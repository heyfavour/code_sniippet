import asyncio
import threading
import time
import uuid
import async_timeout
from types import SimpleNamespace
from typing import TYPE_CHECKING, Awaitable, NoReturn, Optional, Union

from aioredis.exceptions import LockError, LockNotOwnedError

if TYPE_CHECKING: from aioredis import Redis


class RLock:
    """
    可重入锁
    """
    lua_release = None
    lua_extend = None
    lua_acquire = None
    # KEYS[1] - lock name
    # ARGV[1] - token
    # ARGV[2] - TTL
    LUA_ACQUIRE_SCRIPT = """
        if (redis.call('exists', KEYS[1]) == 0) then
            redis.call('hincrby', KEYS[1], ARGV[1], 1)
            redis.call('pexpire', KEYS[1], ARGV[2])
            return 1
        end
        if (redis.call('hexists', KEYS[1], ARGV[1]) == 1) then
            redis.call('hincrby', KEYS[1], ARGV[1], 1)
            redis.call('pexpire', KEYS[1], ARGV[2])
            return 1
        end
        return 0
    """

    # KEYS[1] - lock name
    # ARGV[1] - token
    # ARGV[2] - TTL
    # ARGV[3] - 0 add ttl 1 replace ttl
    LUA_EXTEND_SCRIPT = """
        -- 未拥有锁
        if (redis.call('hexists', KEYS[1], ARGV[1]) == 0) then
            return 0
        end
        -- 剩余时间  持久化返回-1 不存在返回 -2
        local expiration = redis.call('pttl', KEYS[1])
        -- 持久化返回-1 不存在返回 -2
        if expiration < 0 then
            return 0
        end
        
        local newttl = ARGV[2]
        if ARGV[3] == "0" then
            newttl = ARGV[2] + expiration
        end
        -- 续命
        return redis.call('pexpire', KEYS[1], newttl)
    """

    # KEYS[1] - lock name
    # KEYS[2] - channel name
    # ARGV[1] - token
    # ARGV[2] - milliseconds
    LUA_RELEASE_SCRIPT = """
        -- 检查锁是否是自己的
        if (redis.call('hexists', KEYS[1], ARGV[1]) == 0) then
            return 0
        end
        local counter = redis.call('hincrby', KEYS[1], ARGV[1], -1)
        if (counter > 0) then
            redis.call('pexpire', KEYS[1], ARGV[2])
            return 2
        else
            redis.call('del', KEYS[1])
            redis.call('publish', KEYS[2], ARGV[1])
            return 1
        end
        return 0
    """

    def __init__(
            self,
            redis: "Redis",
            key: Union[str, bytes, memoryview],
            token: str = None,
            expiration: Optional[float] = None,
            blocking: bool = True,
            blocking_timeout: Optional[float] = None,
            thread_local: bool = True,
            channel: str = None
    ):
        # hset key token count
        self.redis = redis
        self.key = key
        self.expiration = expiration
        self.token = token
        self.blocking = blocking
        self.blocking_timeout = blocking_timeout
        self.thread_local = bool(thread_local)
        self.local = threading.local() if self.thread_local else SimpleNamespace()
        self.local.token = None
        self.channel = channel if channel else f"redisson_lock__channel:{key}"
        self.register_scripts()

    def register_scripts(self):
        cls = self.__class__
        client = self.redis
        if cls.lua_acquire is None:
            cls.lua_acquire = client.register_script(cls.LUA_ACQUIRE_SCRIPT)
        if cls.lua_release is None:
            cls.lua_release = client.register_script(cls.LUA_RELEASE_SCRIPT)
        if cls.lua_extend is None:
            cls.lua_extend = client.register_script(cls.LUA_EXTEND_SCRIPT)

    async def __aenter__(self):
        if await self.acquire():
            return self
        raise LockError("Unable to acquire lock within the time specified")

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.release()

    async def acquire(
            self,
            blocking: Optional[bool] = None,
            blocking_timeout: Optional[float] = None,
            token: Optional[Union[str, bytes]] = None,
    ):
        """
        获取锁
        blocking - 是否阻塞等待 True 等待 False 加锁失败立即返回
        blocking_timeout - 阻塞时长
        token - uuid
        """
        if token:
            token = token
        elif self.token:
            token = self.token
        elif self.local.token:
            token = self.local.token
        else:
            token = uuid.uuid1().hex
        if blocking is None: blocking = self.blocking
        if blocking_timeout is None: blocking_timeout = self.blocking_timeout

        # 加锁—>阻塞等待->加锁
        if await self.do_acquire(token):
            self.local.token = token
            return True
        if not blocking: return False
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(self.channel)
        try:
            async with async_timeout.timeout(blocking_timeout):
                while True:
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=blocking_timeout)
                    if message is not None: break
        except asyncio.TimeoutError:
            pass
        if await self.do_acquire(token):
            self.local.token = token
            return True
        return False

    async def do_acquire(self, token: Union[str, bytes]) -> bool:
        expiration = int(self.expiration * 1000)  # convert to milliseconds
        acquire = await self.lua_acquire(
            keys=[self.key],
            args=[token, expiration],
            client=self.redis,
        )
        return bool(acquire)

    async def locked(self) -> bool:
        return await self.redis.hexists(self.key)

    async def owned(self) -> bool:
        stored_key = await self.redis.hgetall(self.key)
        stored_token = stored_key.keys()[0]
        return self.local.token is not None and stored_token == self.local.token

    async def release(self) -> Awaitable[NoReturn]:
        expected_token = self.local.token
        if expected_token is None: raise LockError("Cannot release an unlocked lock")
        return await self.do_release(expected_token)

    async def do_release(self, expected_token: bytes):
        # 0 not owner 1 del key 2 count>1
        expiration = int(self.expiration * 1000)
        _release = await self.lua_release(
            keys=[self.key, self.channel],
            args=[expected_token, expiration],
            client=self.redis
        )
        if not bool(_release):
            raise LockNotOwnedError("Cannot release a lock" " that's no longer owned")
        if _release in {0,1}: self.local.token = None
        return _release

    async def extend(self, additional_time: float, replace_ttl: bool = False) -> Awaitable[bool]:
        """
        续命
        """
        if self.local.token is None: raise LockError("Cannot extend an unlocked lock")
        if self.expiration is None: raise LockError("Cannot extend a lock with no timeout")
        return await self.do_extend(additional_time,replace_ttl)

    async def do_extend(self, additional_time, replace_ttl) -> bool:
        additional_time = int(additional_time * 1000)
        if not bool(
                await self.lua_extend(
                    keys=[self.key],
                    args=[self.local.token, additional_time, replace_ttl and "1" or "0"],
                    client=self.redis,
                )
        ):
            raise LockNotOwnedError("Cannot extend a lock that's" " no longer owned")
        return True


async def demo():
    import aioredis
    redis = aioredis.from_url("redis://192.168.99.20:6379", password="q1w2e3r4T%Y^U&", decode_responses=True, )
    key = "cust_no:product_no"
    token = str(uuid.uuid1())
    print(key, token)
    lock = RLock(redis, key, expiration=1000)
    _lock1 = await lock.acquire(blocking=True, blocking_timeout=3)
    print(_lock1, await redis.hgetall(key))
    _lock2 = await lock.acquire(blocking=True, blocking_timeout=3)
    print(_lock2, await redis.hgetall(key))
    await asyncio.sleep(5)
    print("sleep", await redis.hgetall(key))
    _release1 = await lock.release()
    print(_release1, await redis.hgetall(key))
    await asyncio.sleep(5)
    print("ttl",await redis.pttl(key))
    await lock.extend(10)
    print("ttl",await redis.pttl(key))
    _release2 = await lock.release()
    print(_release1, await redis.hgetall(key))
    time.sleep(2)
    _lock1 = await lock.acquire(blocking=True, blocking_timeout=20)
    print(_lock1, await redis.hgetall(key))
    _release1 = await lock.release()
    print(_release1, await redis.hgetall(key))
    _lock1 = await lock.acquire(blocking=True, blocking_timeout=3)
    print(_lock1, await redis.hgetall(key))
    _release1 = await lock.release()
    print(_release1, await redis.hgetall(key))

if __name__ == '__main__':
    asyncio.run(demo())

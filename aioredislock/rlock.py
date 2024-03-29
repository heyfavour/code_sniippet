import asyncio
import uuid
import async_timeout
import contextvars

from typing import TYPE_CHECKING, Awaitable, NoReturn, Optional, Union
from aioredis.exceptions import LockError, LockNotOwnedError

if TYPE_CHECKING: from aioredis import Redis

_task_local = contextvars.ContextVar("_task_local", default=None)


class SimpleToken:
    def __init__(self):
        self.token = None

    def get(self):
        return self.token

    def set(self, token: str):
        self.token = token


class RLock:
    """
    可重入锁 通过重试轮询acquire
    """
    lua_release = None
    lua_extend = None
    lua_acquire = None
    # KEYS[1] - lock name
    # ARGV[1] - token
    # ARGV[2] - TTL
    # return 0-获取锁失败 1-获取锁成功
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
    # return  0 失败 1成功
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

    # KEYS[1] - lock name --key
    # ARGV[1] - token--token
    # ARGV[2] - milliseconds--释放锁但count>0续命
    # 0失败 1成功 2可重入锁未释放完
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
            sleep: float = 0.1,
            blocking: bool = True,
            blocking_timeout: Optional[float] = None,
            task_local: bool = True,
    ):
        """
        :param redis:redis instance
        :param key: key
        :param token: uuid
        :param expiration: 到期时间
        :param blocking:True wait False 立即返回
        :param blocking_timeout:等待时间
        :param task_local:协程 task 内是否可重入
        :param channel:释放锁只有的广播key
        """
        # hset key token count
        self.redis = redis
        self.key = key
        self.token = token
        self.expiration = expiration
        self.sleep = sleep
        self.blocking = blocking
        self.blocking_timeout = blocking_timeout
        self.task_local = bool(task_local)
        self.local_token = _task_local if self.task_local else SimpleToken()
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
        if await self.acquire(): return self
        raise LockError("Unable to acquire lock within the time specified")

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.release()

    async def acquire(
            self,
            blocking: Optional[bool] = None,
            blocking_timeout: Optional[float] = None,
            token: Optional[Union[str, bytes]] = None,
    ) -> bool:
        """
        获取锁
        blocking - 是否阻塞等待 True 等待 False 加锁失败立即返回
        blocking_timeout - 阻塞时长
        token - uuid
        """
        loop = asyncio.get_event_loop()
        sleep = self.sleep

        if token and self.token and token != self.token: raise Exception("Token conflict")
        if token:
            token = token
        elif self.token:
            token = self.token
        elif self.local_token.get():
            token = self.local_token.get()
        else:
            token = str(uuid.uuid1().hex)
        if blocking is None: blocking = self.blocking
        if blocking_timeout is None: blocking_timeout = self.blocking_timeout

        stop_trying_at = None
        if blocking_timeout is not None: stop_trying_at = loop.time() + blocking_timeout

        while True:
            if await self.do_acquire(token):
                self.local_token.set(token)
                return True
            if not blocking: return False
            next_try_at = loop.time() + sleep
            if stop_trying_at is not None and next_try_at > stop_trying_at:
                return False
            await asyncio.sleep(sleep)

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
        stored_token = list(stored_key.keys())[0]
        local_token = self.local_token.get()
        return local_token is not None and stored_token == local_token

    async def release(self) -> Awaitable[NoReturn]:
        expected_token = self.local_token.get()
        if not expected_token: raise LockError("Cannot release an unlocked lock")
        return await self.do_release(expected_token)

    async def do_release(self, expected_token: bytes) -> int:
        # 0 not owner 1 del key 2 count > 1
        expiration = int(self.expiration * 1000)
        _release = await self.lua_release(
            keys=[self.key],
            args=[expected_token, expiration],
            client=self.redis
        )  # _release -> int
        if not bool(_release): raise LockNotOwnedError(f"Cannot release a lock:{self.key} that's no longer owned")
        if _release in {0, 1}: self.local_token.set(None)
        return _release

    async def extend(self, additional_time: float, replace_ttl: bool = False) -> Awaitable[bool]:
        """
        续命
        """
        if self.local_token.get() is None: raise LockError("Cannot extend an unlocked lock")
        if self.expiration is None: raise LockError("Cannot extend a lock with no timeout")
        return await self.do_extend(additional_time, replace_ttl)

    async def do_extend(self, additional_time, replace_ttl) -> bool:
        additional_time = int(additional_time * 1000)
        if not bool(
                await self.lua_extend(
                    keys=[self.key],
                    args=[self.local_token.get(), additional_time, replace_ttl and "1" or "0"],
                    client=self.redis,
                )
        ):
            raise LockNotOwnedError(f"Cannot extend a lock {self.key} that's no longer owned")
        return True


class RLockPubSub:
    """
    可重入锁 通过发布订阅解除 block wait
    """
    lua_release = None
    lua_extend = None
    lua_acquire = None
    # KEYS[1] - lock name
    # ARGV[1] - token
    # ARGV[2] - TTL
    # return 0-获取锁失败 1-获取锁成功
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
    # return  0 失败 1成功
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

    # KEYS[1] - lock name --key
    # KEYS[2] - channel name--释放锁的广播channel
    # ARGV[1] - token--token
    # ARGV[2] - milliseconds--释放锁但count>0续命
    # 0失败 1成功 2可重入锁未释放完
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
            task_local: bool = True,
            channel: str = None,
    ):
        """
        :param redis:redis instance
        :param key: key
        :param token: uuid
        :param expiration: 到期时间
        :param blocking:True wait False 立即返回
        :param blocking_timeout:等待时间
        :param task_local:协程 task 内是否可重入
        :param channel:释放锁只有的广播key
        """
        # hset key token count
        self.redis = redis
        self.key = key
        self.expiration = expiration
        self.token = token
        self.blocking = blocking
        self.blocking_timeout = blocking_timeout
        self.task_local = bool(task_local)
        self.local_token = _task_local if self.task_local else SimpleToken()
        self.channel = channel if channel else f"redisson_lock_channel:{key}"
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
        if token and self.token and token != self.token: raise Exception("Token conflict")
        if token:
            token = token
        elif self.token:
            token = self.token
        elif self.local_token.get():
            token = self.local_token.get()
        else:
            token = str(uuid.uuid1().hex)
        if blocking is None: blocking = self.blocking
        if blocking_timeout is None: blocking_timeout = self.blocking_timeout

        # 加锁—>阻塞等待->加锁
        if await self.do_acquire(token):
            self.local_token.set(token)
            return True
        if not blocking: return False
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(self.channel)
        try:
            # 这里有一个bug,async_timeout的timeout可能不会触发 所以暂时不推荐
            # there is a bug here!async_timeout's timeout sometimes not work！Not recommend!
            async with async_timeout.timeout(blocking_timeout):
                while True:
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)
                    if message is not None: break
        except asyncio.TimeoutError:
            pass
        if await self.do_acquire(token):
            self.local_token.set(token)
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
        local_token = self.local_token.get()
        return local_token is not None and stored_token == local_token

    async def release(self) -> Awaitable[NoReturn]:
        expected_token = self.local_token.get()
        if not expected_token: raise LockError("Cannot release an unlocked lock")
        return await self.do_release(expected_token)

    async def do_release(self, expected_token: bytes) -> int:
        # 0 not owner 1 del key 2 count>1
        expiration = int(self.expiration * 1000)
        _release = await self.lua_release(
            keys=[self.key, self.channel],
            args=[expected_token, expiration],
            client=self.redis
        )  # ->int
        if not bool(_release): raise LockNotOwnedError(f"Cannot release a lock:{self.key} that's no longer owned")
        if _release in {0, 1}: self.local_token.set(None)
        return _release

    async def extend(self, additional_time: float, replace_ttl: bool = False) -> Awaitable[bool]:
        """
        续命
        """
        if self.local_token.get() is None: raise LockError("Cannot extend an unlocked lock")
        if self.expiration is None: raise LockError("Cannot extend a lock with no timeout")
        return await self.do_extend(additional_time, replace_ttl)

    async def do_extend(self, additional_time, replace_ttl) -> bool:
        additional_time = int(additional_time * 1000)
        if not bool(
                await self.lua_extend(
                    keys=[self.key],
                    args=[self.local_token.get(), additional_time, replace_ttl and "1" or "0"],
                    client=self.redis,
                )
        ):
            raise LockNotOwnedError(f"Cannot extend a lock {self.key} that's no longer owned")
        return True

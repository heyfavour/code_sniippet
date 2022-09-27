import asyncio
import uuid

from typing import TYPE_CHECKING, Awaitable, NoReturn, Optional, Union
from aioredis.exceptions import LockError, LockNotOwnedError

if TYPE_CHECKING: from aioredis import Redis


class Lock:
    """
    分布式锁 不可重入 所以不进行线程/协程隔离
    """
    lua_release = None
    lua_extend = None

    # KEYS[1] - lock name
    # ARGV[1] - token
    # return 1 if the lock was released, otherwise 0
    LUA_RELEASE_SCRIPT = """
        local token = redis.call('get', KEYS[1])
        if not token or token ~= ARGV[1] then
            return 0
        end
        redis.call('del', KEYS[1])
        return 1
    """

    # KEYS[1] - lock name
    # ARGV[1] - token
    # ARGV[2] - additional milliseconds
    # ARGV[3] - "0" if the additional time should be added to the lock's
    #           existing ttl or "1" if the existing ttl should be replaced
    # return 1 if the locks time was extended, otherwise 0
    LUA_EXTEND_SCRIPT = """
        local token = redis.call('get', KEYS[1])
        if not token or token ~= ARGV[1] then
            return 0
        end
        local expiration = redis.call('pttl', KEYS[1])
        if not expiration then
            expiration = 0
        end
        if expiration < 0 then
            return 0
        end

        local newttl = ARGV[2]
        if ARGV[3] == "0" then
            newttl = ARGV[2] + expiration
        end
        redis.call('pexpire', KEYS[1], newttl)
        return 1
    """

    def __init__(
            self,
            redis: "Redis",
            key: Union[str, bytes, memoryview],
            token: str = None,
            expiration: Optional[float] = 1.0,
            sleep: float = 0.1,
            blocking: bool = True,
            blocking_timeout: Optional[float] = None,
            task_local: bool = True,
    ):
        self.redis = redis
        self.key = key
        self.token = token
        self.expiration = expiration
        self.sleep = sleep
        self.blocking = blocking
        self.blocking_timeout = blocking_timeout
        self.register_scripts()

    def register_scripts(self):
        cls = self.__class__
        client = self.redis
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
        loop = asyncio.get_event_loop()
        sleep = self.sleep

        if token and self.token and token != self.token: raise Exception("Token conflict")
        token = token or self.token or str(uuid.uuid1().hex)
        if blocking is None: blocking = self.blocking
        if blocking_timeout is None: blocking_timeout = self.blocking_timeout
        stop_trying_at = None
        if blocking_timeout is not None: stop_trying_at = loop.time() + blocking_timeout

        while True:
            if await self.do_acquire(token):
                self.token = token
                return True
            if not blocking: return False
            next_try_at = loop.time() + sleep
            if stop_trying_at is not None and next_try_at > stop_trying_at:
                return False
            await asyncio.sleep(sleep)

    async def do_acquire(self, token: Union[str, bytes]) -> bool:
        expiration = int(self.expiration * 1000)
        acquire = await self.redis.set(self.key, token, nx=True, px=expiration)
        return bool(acquire)

    async def locked(self) -> bool:
        return await self.redis.get(self.key) is not None

    async def owned(self) -> bool:
        stored_token = await self.redis.get(self.key)
        return self.token is not None and stored_token == self.token

    def release(self) -> Awaitable[NoReturn]:
        expected_token = self.token
        if expected_token is None: raise LockError("Cannot release an unlocked lock")
        return self.do_release(expected_token)

    async def do_release(self, expected_token: bytes) -> int:
        """
        :param expected_token:
        :return: 0 failed 1 success
        """
        release = await self.lua_release(keys=[self.key], args=[expected_token], client=self.redis)
        if not bool(release): raise LockNotOwnedError(f"Cannot release a lock:{self.key} that's no longer owned")
        return release

    async def extend(
            self, additional_time: float, replace_ttl: bool = False
    ) -> Awaitable[bool]:
        if self.token is None: raise LockError("Cannot extend an unlocked lock")
        if self.expiration is None: raise LockError("Cannot extend a lock with no timeout")
        return await self.do_extend(additional_time, replace_ttl)

    async def do_extend(self, additional_time, replace_ttl) -> bool:
        additional_time = int(additional_time * 1000)
        if not bool(
                await self.lua_extend(
                    keys=[self.key],
                    args=[self.token, additional_time, replace_ttl and "1" or "0"],
                    client=self.redis,
                )
        ):
            raise LockNotOwnedError(f"Cannot extend a lock {self.key} that's no longer owned")
        return True

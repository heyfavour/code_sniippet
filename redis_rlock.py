import asyncio
import threading
import uuid
from types import SimpleNamespace
from typing import TYPE_CHECKING, Awaitable, NoReturn, Optional, Union

from aioredis.exceptions import LockError, LockNotOwnedError

if TYPE_CHECKING:from aioredis import Redis


class RLock:
    """
    可重入锁
    """
    lua_release = None
    lua_extend = None
    lua_reacquire = None

    # KEYS[1] - lock name
    # ARGV[1] - token
    # ARGV[2] - TTL
    LUA_REACQUIRE_SCRIPT = """
        -- 判断锁锁是否存在
        -- 不存在则设置key field=token count=count+1 return 1
        if (redis.call('exists', KEYS[1]) == 0) then
            redis.call('hincrby', KEYS[1], ARGV[1], 1)
            redis.call('pexpire', KEYS[1], ARGV[2])
            return 1
        end
        -- 存在 检查是否拥有锁 拥有锁 设置count=count+1 return 1
        if (redis.call('hexists', KEYS[1], ARGV[1]) == 1) then
            redis.call('hincrby', KEYS[1], ARGV[1], 1)
            redis.call('pexpire', KEYS[1], ARGV[2])
            return 1
        end
        -- 不拥有锁 return 0
        return 0
    """

    # KEYS[1] - lock name
    # ARGV[1] - token
    # ARGV[2] - TTL
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
        -- 续命
        return redis.call('', KEYS[1], ARGV[2])
    """

    # KEYS[1] - lock name
    # KEYS[2] - channel_name
    # ARGV[1] - token
    # ARGV[2] - milliseconds
    LUA_RELEASE_SCRIPT = """
        -- 检查锁是否是自己的
        if (redis.call('hexists', KEYS[1], ARGV[1]) == 0) then
            return 0
        end
        local counter = redis.call('hincrby', KEYS[1], ARGV[3], -1)
        if (counter > 0) then
            redis.call('pexpire', KEYS[1], ARGV[2])
            return 0
        else
            redis.call('del', KEYS[1])
            #redis.call('publish', KEYS[2], ARGV[1])
            return 1
        end
        return 0
    """
    # KEYS[1] - lock name
    # KEYS[1] - channel_name
    LUA_FORCE_SCRIPT = """
        if (redis.call('del', KEYS[1]) == 1) then
            redis.call('publish', KEYS[2], ARGV[1])
            return 1
        else
            return 0 "
        end
    """

    def __init__(
        self,
        redis: "Redis",
        name: Union[str, bytes, memoryview],
        timeout: Optional[float] = None,
        sleep: float = 0.1,
        blocking: bool = True,
        blocking_timeout: Optional[float] = None,
        thread_local: bool = True,
    ):
        self.redis = redis
        self.name = name
        self.timeout = timeout
        self.sleep = sleep
        self.blocking = blocking
        self.blocking_timeout = blocking_timeout
        self.thread_local = bool(thread_local)
        self.local = threading.local() if self.thread_local else SimpleNamespace()
        self.local.token = None
        self.register_scripts()

    def register_scripts(self):
        cls = self.__class__
        client = self.redis
        if cls.lua_release is None:
            cls.lua_release = client.register_script(cls.LUA_RELEASE_SCRIPT)
        if cls.lua_extend is None:
            cls.lua_extend = client.register_script(cls.LUA_EXTEND_SCRIPT)
        if cls.lua_reacquire is None:
            cls.lua_reacquire = client.register_script(cls.LUA_REACQUIRE_SCRIPT)
        if cls.lua_force is None:
            cls.lua_reacquire = client.register_script(cls.LUA_FORCE_SCRIPT)

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
        Use Redis to hold a shared, distributed lock named ``name``.
        Returns True once the lock is acquired.

        If ``blocking`` is False, always return immediately. If the lock
        was acquired, return True, otherwise return False.

        ``blocking_timeout`` specifies the maximum number of seconds to
        wait trying to acquire the lock.

        ``token`` specifies the token value to be used. If provided, token
        must be a bytes object or a string that can be encoded to a bytes
        object with the default encoding. If a token isn't specified, a UUID
        will be generated.
        """
        loop = asyncio.get_event_loop()
        sleep = self.sleep
        if token is None:
            token = uuid.uuid1().hex.encode()
        else:
            encoder = self.redis.connection_pool.get_encoder()
            token = encoder.encode(token)
        if blocking is None:
            blocking = self.blocking
        if blocking_timeout is None:
            blocking_timeout = self.blocking_timeout
        stop_trying_at = None
        if blocking_timeout is not None:
            stop_trying_at = loop.time() + blocking_timeout
        while True:
            if await self.do_acquire(token):
                self.local.token = token
                return True
            if not blocking:
                return False
            next_try_at = loop.time() + sleep
            if stop_trying_at is not None and next_try_at > stop_trying_at:
                return False
            await asyncio.sleep(sleep)

    async def do_acquire(self, token: Union[str, bytes]) -> bool:
        if self.timeout:
            # convert to milliseconds
            timeout = int(self.timeout * 1000)
        else:
            timeout = None
        if await self.redis.set(self.name, token, nx=True, px=timeout):
            return True
        return False

    async def locked(self) -> bool:
        """
        Returns True if this key is locked by any process, otherwise False.
        """
        return await self.redis.get(self.name) is not None

    async def owned(self) -> bool:
        """
        Returns True if this key is locked by this lock, otherwise False.
        """
        stored_token = await self.redis.get(self.name)
        # need to always compare bytes to bytes
        # TODO: this can be simplified when the context manager is finished
        if stored_token and not isinstance(stored_token, bytes):
            encoder = self.redis.connection_pool.get_encoder()
            stored_token = encoder.encode(stored_token)
        return self.local.token is not None and stored_token == self.local.token

    def release(self) -> Awaitable[NoReturn]:
        """Releases the already acquired lock"""
        expected_token = self.local.token
        if expected_token is None:
            raise LockError("Cannot release an unlocked lock")
        self.local.token = None
        return self.do_release(expected_token)

    async def do_release(self, expected_token: bytes):
        if not bool(
            await self.lua_release(
                keys=[self.name], args=[expected_token], client=self.redis
            )
        ):
            raise LockNotOwnedError("Cannot release a lock" " that's no longer owned")

    def extend(
        self, additional_time: float, replace_ttl: bool = False
    ) -> Awaitable[bool]:
        """
        Adds more time to an already acquired lock.

        ``additional_time`` can be specified as an integer or a float, both
        representing the number of seconds to add.

        ``replace_ttl`` if False (the default), add `additional_time` to
        the lock's existing ttl. If True, replace the lock's ttl with
        `additional_time`.
        """
        if self.local.token is None:
            raise LockError("Cannot extend an unlocked lock")
        if self.timeout is None:
            raise LockError("Cannot extend a lock with no timeout")
        return self.do_extend(additional_time, replace_ttl)

    async def do_extend(self, additional_time, replace_ttl) -> bool:
        additional_time = int(additional_time * 1000)
        if not bool(
            await self.lua_extend(
                keys=[self.name],
                args=[self.local.token, additional_time, replace_ttl and "1" or "0"],
                client=self.redis,
            )
        ):
            raise LockNotOwnedError("Cannot extend a lock that's" " no longer owned")
        return True

    def reacquire(self) -> Awaitable[bool]:
        """
        Resets a TTL of an already acquired lock back to a timeout value.
        """
        if self.local.token is None:
            raise LockError("Cannot reacquire an unlocked lock")
        if self.timeout is None:
            raise LockError("Cannot reacquire a lock with no timeout")
        return self.do_reacquire()

    async def do_reacquire(self) -> bool:
        timeout = int(self.timeout * 1000)
        if not bool(
            await self.lua_reacquire(
                keys=[self.name], args=[self.local.token, timeout], client=self.redis
            )
        ):
            raise LockNotOwnedError("Cannot reacquire a lock that's" " no longer owned")
        return True



LUA_REACQUIRE_SCRIPT = """
    local token = redis.call('get', KEYS[1])
    if not token or token ~= ARGV[1] then
        return 0
    end
    redis.call('pexpire', KEYS[1], ARGV[2])
    return 1
"""

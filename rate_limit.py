from functools import wraps
from math import floor

import time
import sys
import asyncio

now = time.monotonic if hasattr(time, 'monotonic') else time.time


class RateLimitException(Exception):
    def __init__(self, message, period_remaining):
        super(RateLimitException, self).__init__(message)
        self.period_remaining = period_remaining


class RateLimitDecorator(object):
    def __init__(self, calls=15, period=900, clock=now, raise_on_limit=True):
        self.clamped_calls = max(1, min(sys.maxsize, floor(calls)))
        self.period = period
        self.clock = clock
        self.raise_on_limit = raise_on_limit

        self.last_reset = clock()
        self.num_calls = 0

        self.lock = asyncio.Lock()

    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kargs):
            async with self.lock:
                period_remaining = self.__period_remaining()
                if period_remaining <= 0:
                    self.num_calls = 0
                    self.last_reset = self.clock()
                self.num_calls += 1
                if self.num_calls > self.clamped_calls:
                    if self.raise_on_limit:
                        raise RateLimitException('too many calls', period_remaining)
                    return
            result = await func(*args, **kargs)
            return result

        return wrapper

    def __period_remaining(self):
        elapsed = self.clock() - self.last_reset
        return self.period - elapsed


def sleep_and_retry(func):
    @wraps(func)
    async def wrapper(*args, **kargs):
        while True:
            try:
                return await func(*args, **kargs)
            except RateLimitException as exception:
                await asyncio.sleep(exception.period_remaining)

    return wrapper


limits = RateLimitDecorator

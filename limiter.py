from functools import wraps
from math import floor
import time
import sys
import threading

from ratelimit.exception import RateLimitException

now = time.monotonic if hasattr(time, 'monotonic') else time.time

class RateLimitDecorator(object):
    def __init__(self, calls=15, period=900, clock=now, raise_on_limit=True):
        self.clamped_calls = max(1, min(sys.maxsize, floor(calls)))
        self.period = period
        self.clock = clock
        self.raise_on_limit = raise_on_limit

        self.last_reset = clock()
        self.num_calls = 0
        self.lock = threading.RLock()

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kargs):
            with self.lock:
                period_remaining = self.__period_remaining() #调用时查看剩余周期
                if period_remaining <= 0:
                    self.num_calls = 0
                    self.last_reset = self.clock()

                # Increase the number of attempts to call the function.
                self.num_calls += 1

                # If the number of attempts to call the function exceeds the
                # maximum then raise an exception.
                if self.num_calls > self.clamped_calls:
                    if self.raise_on_limit:
                        raise RateLimitException('too many calls', period_remaining)
                    return

            return func(*args, **kargs)
        return wrapper

    def __period_remaining(self):
        '''
        Return the period remaining for the current rate limit window.

        :return: The remaing period.
        :rtype: float
        '''
        elapsed = self.clock() - self.last_reset
        return self.period - elapsed
"""
https://www.bilibili.com/read/cv14867248
https://www.bilibili.com/read/cv14883668
signal.SIGALRM 只能用于linux
"""
import datetime
import os
import time
import signal
import errno
from functools import wraps


class TimeoutError(Exception):
    def __init__(self, error_message):
        raise Exception(error_message)


def timeout(seconds=30, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)  # 0成功
            return result

        return wraps(func)(wrapper())

    return decorator


@timeout(2, "测试超时")
def test_timeout():
    print(datetime.datetime.now())
    time.sleep(5)
    print(datetime.datetime.now())


if __name__ == '__main__':
    test_timeout()

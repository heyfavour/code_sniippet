import random
import threading
import contextvars, asyncio


def thread_run_next():
    for i in range(100):
        local.num = local.num + 1


def thread_run():
    local.num = random.randint(1,10)
    for i in range(100):
        local.num = local.num + 1
    thread_run_next()
    print(local.num)


def threading_run():
    global local
    local = threading.local()
    t1 = threading.Thread(target=thread_run)
    t2 = threading.Thread(target=thread_run)
    t1.start()
    t2.start()


async def two(i):
    # lock = contextvars.ContextVar("lock")
    for i in range(100):
        lock.set(lock.get() + 1)


async def one(i):
    # lock = contextvars.ContextVar("lock")
    lock.set(i)
    await two(i)
    for i in range(100):
        lock.set(lock.get() + 1)
    print(lock.get())


async def async_run():
    global lock
    lock = contextvars.ContextVar("lock")
    tasks = [one(i) for i in range(5)]
    await asyncio.gather(*tasks)


if __name__ == '__main__':
    threading_run()
    #asyncio.run(async_run())

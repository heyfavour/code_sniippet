import asyncio
import datetime
from rate_limit import limits, sleep_and_retry


async def time_func(item):
    await asyncio.sleep(1)


class AsyncPool():
    def __init__(self):
        self.sema = asyncio.Semaphore(3)
        # self.lock = asyncio.Lock()

    def get_deal_list(self):
        deal_list = [i for i in range(10)]
        return deal_list

    async def sema_lock_deal(self, item):
        async with self.sema:
            await self.deal_item(item)

    @sleep_and_retry
    @limits(calls=4, period=1)
    async def deal_item(self, item):
        print(item, "deal start", datetime.datetime.now())
        await time_func(item)
        print(item, "deal end", datetime.datetime.now())


def async_pool():
    pool = AsyncPool()
    items = pool.get_deal_list()
    task = [pool.sema_lock_deal(i) for i in items]
    loop = asyncio.new_event_loop()
    loop.run_until_complete(asyncio.wait(task))


if __name__ == "__main__":
    async_pool()

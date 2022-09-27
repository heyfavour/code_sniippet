import logging
import time
import aioredis
import asyncio
import uuid

from rlock import RLock

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(lineno)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_lock_func2(key, i):
    redis = aioredis.from_url("redis://49.235.242.224:6379", password="wzx940516", decode_responses=True, )

    lock = RLock(redis, key, expiration=30)
    _lock = await lock.acquire(blocking=True, blocking_timeout=2)
    logger.info(f"[{i}] {key} {_lock} {await redis.hgetall(key)}=={lock.local_token.get()} 第二次获取锁")
    logger.info(f"[{i}] {key} {await lock.owned()}  第二次是否拥有锁")
    await asyncio.sleep(3)
    logger.info(f"[{i}] sleep --->:{await redis.hgetall(key)} 第二次锁后sleep")
    async with RLock(redis, key, expiration=30, blocking_timeout=2) as lock:
        await asyncio.sleep(0.1*i)
        logger.info(f"[{i}] {key} {lock.local_token.get()} {await redis.hgetall(key)} 第三次锁")

    try:
        async with RLock(redis, key,token="error_token", expiration=30,blocking_timeout=2) as lock:
            logger.info(f"[{i}] {key} {lock.token} {await redis.hgetall(key)} 第四次锁")
    except aioredis.exceptions.LockError:
        logger.info(f"[{i}] {key}  第四次锁异常结束")

    if _lock:
        _release = await lock.release()
        logger.info(f"[{i}] {key} {_release} {await redis.hgetall(key)} 释放第二次锁")

async def test_lock_func3(key, i):
    redis = aioredis.from_url("redis://49.235.242.224:6379", password="wzx940516", decode_responses=True, )

    try:
        async with RLock(redis, key, expiration=30,blocking_timeout=2) as lock:
            logger.info(f"[{i}] {key} {lock.local_token.get()} {await redis.hgetall(key)} 第五次锁")
            await asyncio.sleep(5)
            inner_lock = RLock(redis, key, expiration=30)
            _lock = await inner_lock.acquire(blocking=False)
            logger.info(f"[{i}] {key} {_lock} {inner_lock.local_token.get()} {await redis.hgetall(key)} 第六次锁成功")
            _release = await inner_lock.release()
            logger.info(f"[{i}] {key} {_lock} {inner_lock.local_token.get()} {await redis.hgetall(key)} 第六次放锁成功")
    except aioredis.exceptions.LockError:
        logger.info(f"[{i}] {key}  第五次锁上锁异常")
    logger.info(f"[{i}] {key} {await redis.hgetall(key)}  第五次锁结束")



async def test_lock_func(i):
    redis = aioredis.from_url("redis://49.235.242.224:6379", password="wzx940516", decode_responses=True, )
    key = f"cust:product:{i % 2}"
    try:
        token = str(uuid.uuid1().hex)
        logger.info(f"[{i}] {key}:{token}")
        lock = RLock(redis, key,token, expiration=30)
        # lock = RLock(redis, key, expiration=30)
        _lock = await lock.acquire(blocking=True, blocking_timeout=2)
        logger.info(f"[{i}] {key} {_lock} {await redis.hgetall(key)} 第一次获取锁")
        logger.info(f"[{i}] {key} {await lock.owned()}  第一次是否拥有锁")
        await test_lock_func2(key, i)
        await asyncio.sleep(3)
        # logger.info(f"[{i}] {key} {await redis.hgetall(key)} ttl:{await redis.pttl(key)} 查看剩余时间")
        # await lock.extend(10)  #
        # logger.info(f"[{i}] {key} {await redis.hgetall(key)} ttl:{await redis.pttl(key)} 查看剩余时间")
        if _lock:
            _release = await lock.release()
            logger.info(f"[{i}] {key} {_release} {await redis.hgetall(key)} 释放第一次锁")
        await test_lock_func3(key,i)
    except Exception as exc:
        import traceback
        traceback.print_exception(exc)
        logger.info(f"[{i}]{key}{str(exc)}")


async def task_run():
    # tasks = [test_lock_func(i) for i in range(1)]
    tasks = [test_lock_func(i) for i in range(3)]
    await asyncio.gather(*tasks)


if __name__ == '__main__':
    asyncio.run(task_run())

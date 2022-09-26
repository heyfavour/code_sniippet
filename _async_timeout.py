import asyncio
import datetime

import async_timeout

async def main():
    try:
        start_time = datetime.datetime.now()
        async with async_timeout.timeout(2):
            async with async_timeout.timeout(2):
                async with async_timeout.timeout(2):
                    await asyncio.sleep(5)  # io操作
                    print('协程执行完成')
    except asyncio.TimeoutError:
        end_time = datetime.datetime.now()
        print(f'协程执行超时{end_time-start_time}')


if __name__ == '__main__':
    asyncio.run(main())
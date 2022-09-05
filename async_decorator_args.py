import asyncio
from functools import wraps


def _decorator(demo_arg):
    def decorator(func):
        @wraps(func)
        async def wrapper(data, session, logger):
            data = {
                "demo_arg": demo_arg,
                "data": data
            }
            data = await func(data, session, logger)
            return data["data"]

        return wrapper

    return decorator


@_decorator("XXXX")
async def _func(data, session, logger):
    print(data)
    # do some things to get data
    data = {"data": {"1": "2", "2": "4"}}
    return data


async def run():
    data = {"demo": "start"}
    result = await _func(data, None, None)
    print(result)
    return result


if __name__ == '__main__':
    asyncio.run(run())

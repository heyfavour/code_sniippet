from functools import wraps


def _decorator(demo_arg):
    def decorator(func):
        @wraps(func)
        def wrapper(data, session, logger):
            data = {
                "demo_arg": demo_arg,
                "data": data
            }
            data = func(data, session, logger)
            return data["data"]

        return wrapper

    return decorator


@_decorator("XXXX")
def _func(data, session, logger):
    print(data)
    # do some things to get data
    data = {"data": {"1": "2", "2": "4"}}
    return data


if __name__ == '__main__':
    data = {"demo": "start"}
    print(_func(data, None, None))

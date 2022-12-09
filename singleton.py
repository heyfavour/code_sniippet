#不推荐会重复执行init 会重新加载一次init的数据
import threading


class Singleton:
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(Singleton, "_instance"):
            with Singleton._instance_lock:
                if not hasattr(Singleton, "_instance"): Singleton._instance = super().__new__(cls)
        return Singleton._instance

    def __init__(self):
        pass

#推荐
import threading


def Singleton(cls):
    _instance = {}
    lock = threading.Lock()

    def _singleton(*args, **kargs):
        if cls not in _instance:
            with lock:
                if cls not in _instance:
                    _instance[cls] = cls(*args, **kargs)
        return _instance[cls]

    return _singletons
   
@Singleton
class Demo():
    def __init__(self):
        pass
        
func = Singleton(Demo)
func()

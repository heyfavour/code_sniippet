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

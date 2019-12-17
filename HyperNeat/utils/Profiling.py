import time
import pandas as pd

exec_times = {}


def get_exec_times():
    return exec_times

def get_exec_times_pd():
    l = [(name, sum/n, sum, n) for name,(n, sum) in exec_times.items() ]
    return pd.DataFrame(l, columns=['name', 'avg_time', 'total_time', 'N'])

class profile_section:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()


    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.name in exec_times:
            n, sum = exec_times[self.name]
        else:
            n, sum = 0, 0
        exec_times[self.name] = (n+1, sum + time.time()-self.t)



def profile_func(_name):
    def wrap(func):
        name = func.__name__ if _name is None else _name
        def wrapper(*args,**kwargs):
            with profile_section(name):
                return func(*args, **kwargs)
        return wrapper
    return wrap



if __name__ == '__main__':
    @profile_func("name")
    def time_me(test_arg):
        print(test_arg)
       # time.sleep(1.23)

    time_me("TEST")
    time_me("TEST")
    print(get_exec_times_pd())






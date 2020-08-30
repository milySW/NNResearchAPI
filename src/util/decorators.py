from functools import wraps
from time import time


def timespan(process_name):
    def timing(f):
        @wraps(f)
        def wrap(*args, **kw):
            ts = time()
            result = f(*args, **kw)
            te = time()
            print(f"{process_name} took: {round(te - ts, 4)} sec")
            return result

        return wrap

    return timing

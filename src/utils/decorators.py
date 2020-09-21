import sys
from functools import wraps
from time import time


def timespan(process_name: str):
    """
    Decorator based on `High Performance Python - Practical Performant
    Programming for Humans` by Micha Gorelicki & Ian Ozsvald.
    Chapeter 2: Profiling to Find Bottlenecks, example 2-5.
    """

    def timing(f):
        @wraps(f)
        def wrap(*args, **kw):
            ts = time()
            result = f(*args, **kw)
            te = time()
            sys.stdout.write(f"{process_name} took: {round(te - ts, 4)} sec")
            return result

        return wrap

    return timing

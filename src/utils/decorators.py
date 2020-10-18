import sys

from functools import wraps
from time import time

import numpy as np
import pandas as pd


def timespan(name: str):
    """
    Decorator based on `High Performance Python - Practical Performant
    Programming for Humans` by Micha Gorelicki & Ian Ozsvald.
    Chapeter 2: Profiling to Find Bottlenecks, example 2-5.
    """

    def timing(f):
        @wraps(f)
        def wrap(*args, **kwargs):
            ts = time()
            result = f(*args, **kwargs)
            te = time()

            seconds = round(te - ts, 4)
            sys.stdout.write(f"\n{name} took: {seconds} sec\n")

            if result:
                root = np.atleast_1d(result)[0]
                path = root / "time.csv"
                header = not path.is_file()

                df = pd.DataFrame({"operation": [name], "time[s]": [seconds]})
                df.to_csv(path, mode="a", index=False, header=header)

            return result

        return wrap

    return timing

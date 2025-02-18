import functools
from typing import Callable


class Retry:
    def __init__(self, retries: int):
        self.retries = retries

    def __call__(self, func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(self.retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == self.retries - 1:
                        raise
        return wrapper
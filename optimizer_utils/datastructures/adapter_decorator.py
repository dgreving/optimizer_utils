from .parameter import Parameter
import functools


def enable_parameters(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        new_args = [
            x.value if isinstance(x, Parameter)
            else x for x in args]
        new_kwargs = {
            key: (val.value if isinstance(val, Parameter) else val)
            for key, val in kwargs.items()
        }
        return func(*new_args, **new_kwargs)
    return inner

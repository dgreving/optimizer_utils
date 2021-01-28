import pytest

from xray_diffraction.datastructures.adapter_decorator import enable_parameters
from xray_diffraction.datastructures.parameter import Parameter


@enable_parameters
def rnd_func(a, b):
    """rnd_func docstring"""
    return a, b


@enable_parameters
def rnd_func_2(*args):
    return args


class Test_enable_parameters:
    def test_works_with_positional_args(self):
        _float, parameter = 42., Parameter('test_para', 1)
        a, b = rnd_func(_float, parameter)
        assert a == 42. and b == 1

    def test_works_with_arg_unpacking(self):
        _float, p1 = 42., Parameter('first', 1)
        p2 = Parameter('second', 2, coupler=('additive', p1))
        args = rnd_func_2(*[_float, p1, p2])
        expected = (42., 1, 1+2)
        assert all(x == e for x, e in zip(args, expected))

    def test_name_and_doc_is_kept(self):
        assert rnd_func.__name__.strip() == 'rnd_func'
        assert rnd_func.__doc__ == 'rnd_func docstring'


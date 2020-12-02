import pytest

from xray_diffraction.datastructures.parameter_controller import (
    ParameterController
)


@pytest.fixture
def pc():
    return ParameterController(name='test_controller')


@pytest.fixture
def paras():
    from xray_diffraction.datastructures.parameter import Parameter
    p1 = Parameter(name='plain', value=1)
    p2 = Parameter(name='coupled', value=2, coupler=('additive', p1))
    return p1, p2


def test_init():
    import collections
    pc = ParameterController(name='test', para_suffix='test_suffix')
    assert isinstance(pc, ParameterController)
    assert pc.name == 'test' and pc.para_suffix == 'test_suffix'
    assert pc.collection is None


def test_add_and_get_parameters(pc, paras):
    p1, p2 = paras[:2]
    pc.add_parameters(*paras)
    assert pc.get_parameter('plain') is p1
    assert pc.get_parameter('coupled') is p2

def test_parameters_accessed_wo_suffix():
    pass

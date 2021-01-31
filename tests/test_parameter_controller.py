import pytest

from xray_diffraction.datastructures.parameter_controller import (
    ParameterController
)


@pytest.fixture
def pc():
    return ParameterController(name='test_controller', suffix='sfx')


@pytest.fixture
def paras():
    from xray_diffraction.datastructures.parameter import Parameter
    from xray_diffraction.datastructures.parameter import ReferenceParameter
    p1 = Parameter(name='plain', value=1)
    p2 = Parameter(name='coupled', value=2, coupler=('additive', p1))
    p3 = ReferenceParameter(name='identical', references=p2)
    return p1, p2, p3


def test_init():
    pc = ParameterController(name='test', suffix='test_suffix')
    assert isinstance(pc, ParameterController)
    assert pc.name == 'test' and pc.suffix == 'test_suffix'


def test_add_and_get_parameters(pc, paras):
    p1, p2 = paras[:2]
    pc.add(*paras)
    assert pc.get('plain') is p1  # note parameter access w/o suffix
    assert pc.get('coupled') is p2
    with pytest.raises(KeyError):
        pc.get('non_existent_parameter')


def test_suffixing(pc, paras):
    plain = paras[0]
    pc.add(plain)
    assert pc.get('plain') is plain
    assert pc.get('plain').name == 'plain__sfx'


def test_add_overwriting_behaviour(pc, paras):
    import logging
    from xray_diffraction.datastructures.parameter import Parameter
    plain = paras[0]
    pc.add(*paras)
    same_name_para = Parameter(name='plain', value=5000)
    with pytest.raises(ValueError) as excinfo:
        pc.add(same_name_para)
    msg = 'Different parameter of name "plain" exists already'
    assert msg in str(excinfo.value)


def test_get_value(pc, paras):
    pc.add(*paras)
    assert pc.get_value('coupled') == 1 + 2
    assert pc.get_value('coupled', no_coupling=True) == 2


def test__getitem__(pc, paras):
    pc.add(*paras)
    assert pc['coupled'] is paras[1]
    assert pc['coupled'].value == 1 + 2


def test_update(pc, paras):
    plain, coupled, identical = paras[:3]
    pc.add(*paras)
    assert plain.value == 1
    assert coupled.value == 2 + 1
    assert identical.value == 2 + 1
    pc.update(('plain', 1001))
    assert plain.value == 1001
    pc.update(('plain', 1002), ('coupled', 8))
    assert plain.value == 1002 and coupled.value == 1010
    pc.update(plain=0, coupled=1)
    assert plain.value == 0 and coupled.value == 1
    with pytest.raises(TypeError):
        pc.update(identical=42)
    with pytest.raises(KeyError):
        pc.update(non_existent_key=0)


def test_as_list(pc, paras):
    paras = paras[:3]
    plain, coupled, identical = paras
    for p, fit in zip(paras, [False, True]):
        p.fit = fit
    pc.add(plain, coupled, identical)
    rv = pc.as_list()
    assert all(returned is original for returned, original in zip(rv, paras))
    rv = pc.as_list(only_fitted=True)
    assert all(a is b for a, b in zip(rv, [coupled]))


def test_num_paras(pc, paras):
    paras = paras[:3]  # plain, coupled, identical
    for p, fit in zip(paras, [False, True]):
        p.fit = fit
    pc.add(*paras)
    assert pc.num_paras() == 3
    assert pc.num_paras(only_fitted=True) == 1  # identical doesn't add to total


def test_iter(pc, paras):
    pc.add(*paras[:3])
    names = [name for name in pc]
    assert all(
        a == b for a, b
        in zip(names, ['plain', 'coupled', 'identical'])
    )


def test_merge(pc, paras):
    plain, coupled, identical = paras
    pc2 = ParameterController(name='pc2')
    pc.add(plain, coupled)
    pc2.add(identical)
    pc.merge(pc2)
    assert pc.get('identical') is identical


def test_values(pc, paras):
    coupled = paras[1]
    pc.add(*paras)
    pc['coupled'].fit = True
    all_paras = pc.values()
    fitted_paras = pc.values(only_fitted=True)
    assert all(a is b for a, b in zip(all_paras, paras))
    assert all(a is b for a, b in zip(fitted_paras, [coupled]))


def test_keys_method_only_fitted(pc, paras):
    pc.add(*paras)
    paras[1].fit = True
    all_keys = pc.keys()
    fitted_keys = pc.keys(only_fitted=True)
    all_expected = [para.name for para in paras]
    fitted_expected = [para.name for para in paras if para.fit]
    assert all(a == b for a, b in zip(all_keys, all_expected))
    assert all(a == b for a, b in zip(fitted_keys, fitted_expected))


def test_keys_method_suffixing(pc, paras):
    pc.add(*paras)
    suffixed = [para.name for para in paras]
    unsuffixed = [para.name.rstrip('__sfx') for para in paras]
    assert all(a == b for a, b in zip(pc.keys(suffixed=True), suffixed))
    assert all(a == b for a, b in zip(pc.keys(suffixed=False), unsuffixed))


def test_repr(pc, paras):
    pc.add(*paras)
    representation = repr(pc)
    assert 'Class: ParameterController' in representation
    assert all(p.name in representation for p in paras)

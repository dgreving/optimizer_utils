import pytest
import logging
import sys

import xray_diffraction.datastructures.coupler as _coupler

from xray_diffraction.datastructures.parameter import IParameter
from xray_diffraction.datastructures.parameter import Parameter
from xray_diffraction.datastructures.parameter import ReferenceParameter
from xray_diffraction.datastructures.parameter import ComplexParameter
from xray_diffraction.datastructures.parameter import ScatteringFactorParameter
from xray_diffraction.datastructures.parameter import ParameterGroup


# =============================================================================
# =============================================================================
# =============================================================================


class TestCoupler:
    def test_NoCoupler(self, capsys):
        from xray_diffraction.datastructures.compatability import logger
        streamhandler = logging.StreamHandler(sys.stdout)
        logger.addHandler(streamhandler)

        p = Parameter(name='')
        coupler = _coupler.NoCoupler()
        coupler.couple(p)

        captured = capsys.readouterr()
        assert captured.out == ''

    def test_IdentityCoupler(self):
        from xray_diffraction.datastructures.coupler import IdentityCoupler
        plain = Parameter(name='plain', value=42)
        coupled = Parameter(name='coupled', value=10, coupler=('additive', plain))
        identical = Parameter(name='coupled', coupler=IdentityCoupler(coupled))
        assert coupled.value == 52
        assert identical.value == 52

    def test_ArithmeticCoupler(self):
        base = Parameter(name='base', value=10)
        modifier = Parameter(name='modifier', value=5)
        coupler = _coupler.ArithmeticCoupler(base=base, modifier=modifier)
        coupler._op = '+'
        assert coupler.value == 15

    @pytest.mark.parametrize(
        'coupler,expected',
        [
            (_coupler.AdditiveCoupler, 17),
            (_coupler.SubtractiveCoupler, 3),
            (_coupler.MultiplicativeCoupler, 100),
        ])
    def test_ArithmeticCoupling(self, coupler, expected):
        base = Parameter(name='base', value=10)
        firstmod = Parameter(
            name='first_modifier',
            value=5,
            coupler=coupler(base),
            )
        secondmod = Parameter(
            name='second_modifier',
            value=2,
            coupler=coupler(firstmod),
            )
        assert secondmod.value == expected


# =============================================================================
# =============================================================================
# =============================================================================


class TestParameter:
    @pytest.mark.parametrize(
        'attr',
        ['name', 'value', 'bounds', 'fit', 'coupler', 'bounds_are_relative'])
    def test_has_attr(self, attr):
        p = Parameter(name='test_parameter')
        assert hasattr(p, attr)

    def test_get_and_set_name(self):
        p = Parameter(name='test_parameter')
        assert p.name == 'test_parameter'
        p.name = 'new_name'
        assert p.name == 'new_name'

    def test_get_and_set_value_uncoupled(self):
        p = Parameter(name='', value=50)
        assert p.value == 50
        p.set_value(150)
        assert p.value == 150

    @pytest.mark.parametrize(
        'val,bounds,are_relative,expected',
        [
            (10, [-2, 2], False, [-2, 2]),
            (10, [-2, 2], True, [-20, 20]),
            ]
        )
    def test_get_and_set_bounds(self, val, bounds, are_relative, expected):
        p = Parameter(
            name='',
            value=val,
            bounds=bounds,
            bounds_are_relative=are_relative,
            )
        assert p.bounds[0] == expected[0]
        assert p.bounds[1] == expected[1]

    def test_get_and_set_fit(self):
        p1 = Parameter(name='', fit=True)
        p2 = Parameter(name='', fit=False)
        assert p1.fit and not p2.fit
        p1.fit = False
        assert not p1.fit

    def test_get_and_set_coupled(self):
        b = Parameter(name='base', value=1)
        m = Parameter(
            name='modifier',
            value=2,
            coupler=_coupler.AdditiveCoupler(b),
            )
        assert m.get_value() == 3
        assert m.get_value(no_coupling=True) == 2
        m.set_value(3)
        assert m.get_value() == 4
        assert m.get_value(no_coupling=True) == 3
        b.set_value(5)
        assert m.get_value() == 8
        assert m.get_value(no_coupling=True) == 3

    def test_get_and_set_bounds_are_relative(self):
        p1 = Parameter(name='', bounds_are_relative=True)
        p2 = Parameter(name='', bounds_are_relative=False)
        assert p1.bounds_are_relative and not p2.bounds_are_relative
        p1.bounds_are_relative = False
        assert not p1.bounds_are_relative

    def test_representation(self):
        p = Parameter(
            name='test_parameter',
            value=1,
            bounds=[-1, 2],
            bounds_are_relative=True,
            )
        assert repr(p) == (
            'class: Parameter, '
            'Name: "test_parameter"\t'
            'Value: 1 (1)\t'
            'Bounds: (-1, 2)\t'
            'Fit: False\t'
            'Coupling: class IdentityCoupler, value: 1'
            )


# =============================================================================
# =============================================================================
# =============================================================================


class TestReferenceParameter:
    @pytest.fixture
    def parameters(self):
        plain = Parameter(name='plain', value=42)
        coupled = Parameter(
            name='coupled',
            value=10,
            bounds=(1, 4),
            coupler=('subtractive', plain),
        )
        identical = ReferenceParameter(name='identical', references=coupled)
        return plain, coupled, identical

    def test_identical_coupling(self, parameters):
        _, _, identical = parameters
        assert identical.value == 32

    def test_cannot_modify_IdentityParameter(self, parameters):
        plain, coupled, identical = parameters
        with pytest.raises(TypeError):
            identical.set_value(101)

    def test_get(self, parameters):
        plain, coupled, identical = parameters
        assert identical.get_value() == 32
        assert identical.get_value(no_coupling=True) == 10

    def test_bounds(self, parameters):
        plain, coupled, identical = parameters
        assert identical.bounds == (1, 4)
        with pytest.raises(TypeError):
            identical.bounds = (20, 22)

    def test_fit(self, parameters):
        plain, coupled, identical = parameters
        assert identical.fit is False
        with pytest.raises(TypeError):
            identical.fit = True


class TestComplexParameter:
    @pytest.fixture
    def uncoupled(self):
        a, b = Parameter('real', 1), Parameter('imag', 2)
        c = ComplexParameter('complex', a, b)
        return a, b, c

    @pytest.fixture
    def coupled(self):
        a = Parameter('base', 1)
        b = Parameter('modifier', 2, coupler=_coupler.AdditiveCoupler(a))
        c = ComplexParameter('complex', a, b)
        return a, b, c

    def test_is_parameter(self, uncoupled, coupled):
        _, _, c1 = uncoupled
        _, _, c2 = coupled
        assert isinstance(c1, Parameter)
        assert isinstance(c2, Parameter)

    def test_returns_complex(self, uncoupled, coupled):
        a1, b1, c1 = uncoupled
        a2, b2, c2 = coupled
        assert isinstance(c1.value, complex)
        assert isinstance(c2.value, complex)
        assert c1.value == a1.value + b1.value*1J
        assert c2.value == a2.value + b2.value*1J

    def test_get_value(self, coupled):
        a, b, c = coupled
        assert a.get_value() == 1
        assert b.get_value() == (1 + 2)
        assert c.get_value() == 1 + (1 + 2)*1J
        assert c.get_value(no_coupling=True) == 1 + 2J

    def test_setting_values_works(self, coupled):
        a, b, c = coupled
        assert c.value == 1 + (1+2)*1J
        b.set_value(1000)
        a.set_value(42)
        assert c.value == 42 + (1000+42)*1J
        with pytest.raises(TypeError):
            c.set_value(42 + 1042J)

    def test_bounds(self, uncoupled):
        _, _, c = uncoupled
        with pytest.raises(TypeError):
            c.bounds()


# =============================================================================
# =============================================================================
# =============================================================================


class TestScatteringFactorParameter:

    @pytest.fixture(scope='function')
    def parameters(self):
        real, imag = Parameter('real', 1), Parameter('imag', 3)
        real_mag = Parameter('real_mag', 2)
        imag_mag = Parameter(
            name='imag_mag',
            value=4,
            coupler=_coupler.AdditiveCoupler(real_mag),
            )
        scatt_p = ScatteringFactorParameter(
            'scatt_fac_para', real, imag, real_mag, imag_mag,
            )
        return real, imag, real_mag, imag_mag, scatt_p

    def test_is_parameter(self, parameters):
        scatt_p = parameters[-1]
        assert isinstance(scatt_p, Parameter)

    def test_returns_complex_value(self, parameters):
        scatt_p = parameters[-1]
        assert isinstance(scatt_p.value, complex)
        assert scatt_p.value == (1 + 2) + (3 + (4 + 2))*1J  # note coupling

    def test_sensitive_to_para_modification(self, parameters):
        real, imag, real_mag, imag_mag, scatt_p = parameters
        real.set_value(100)
        assert scatt_p.value == (100 + 2) + (3 + (4 + 2))*1J  # note coupling

    def test_return_modes(self, parameters):
        scatt_p = parameters[-1]
        # test default behaviour
        assert scatt_p.return_mode == 'full'
        # test return_mode 'full'
        scatt_p.return_mode = 'full'
        assert scatt_p.value == (1 + 2) + (3 + 6)*1J
        # test return_mode charge, only non-magnetic contributions
        scatt_p.return_mode = 'charge'
        assert scatt_p.value == (1 + 0) + (3 + 0)*1J
        # test return_mode magn, only magnetic contributions considered
        scatt_p.return_mode = 'magn'
        assert scatt_p.value == (0 + 2) + (0 + 6)*1J
        # test return_mode '+', magnetism contributes positively
        scatt_p.return_mode = '+'
        assert scatt_p.value == (1 + 2) + (3 + 6)*1J
        # test return mode '-', magnetism contributes negatively
        scatt_p.return_mode = '-'
        assert scatt_p.value == (1 - 2) + (3 - 6)*1J
        scatt_p.return_mode = 'unknown_mode'
        with pytest.raises(NameError):
            scatt_p.value


# =============================================================================
# =============================================================================
# =============================================================================


class TestParameterGroup:
    @pytest.fixture
    def parameters(self):
        p1, p2, p3 = Parameter('p1', 1), Parameter('p2', 2), Parameter('p3', 3)
        p4 = Parameter('p4', value=4, coupler=_coupler.AdditiveCoupler(p3))
        group = ParameterGroup('groupName', p1, p2, p3, p4)
        return p1, p2, p3, p4, group

    def test_value(self, parameters):
        group = parameters[-1]
        expected = [1, 2, 3, 3+4]
        assert all(x == y for x, y in zip(group.value, expected))

    def test_get_value(self, parameters):
        group = parameters[-1]
        expected_coupled = [1, 2, 3, 3+4]
        expected_uncoupled = [1, 2, 3, 4]
        assert all(x == y for x, y in zip(group.get_value(), expected_coupled))
        assert all(
            x == y
            for x, y
            in zip(group.get_value(no_coupling=True), expected_uncoupled)
            )

    def test_set_value(self, parameters):
        p1, p2, p3, p4, group = parameters
        with pytest.raises(TypeError):
            group.set_value([4, 5, 6, 7])
        p1.set_value(2)
        p3.set_value(55)
        expected = [2, 2, 55, 55+4]
        assert all(x == y for x, y in zip(group.value, expected))

    def test_bounds(self, parameters):
        group = parameters[-1]
        with pytest.raises(TypeError):
            group.bounds

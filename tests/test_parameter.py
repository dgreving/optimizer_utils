import pytest
import logging
import sys

import xray_diffraction.datastructures.coupler as _coupler

from xray_diffraction.datastructures.parameter import IdentityCoupler
from xray_diffraction.datastructures.parameter import Parameter
from xray_diffraction.datastructures.parameter import ComplexParameter
from xray_diffraction.datastructures.parameter import ScatteringFactorParameter
from xray_diffraction.datastructures.parameter import ParameterGroup


# =============================================================================
# =============================================================================
# =============================================================================


class TestCoupler:
    def test_Identity_Coupler(self, capsys):
        from xray_diffraction.datastructures.compatability import logger
        streamhandler = logging.StreamHandler(sys.stdout)
        logger.addHandler(streamhandler)

        p = Parameter(name='')
        coupler = IdentityCoupler()
        coupler.couple(p)

        captured = capsys.readouterr()
        assert captured.out == ''

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


class TestComplexParameter:
    @pytest.fixture
    def uncoupled(self):
        A, B = Parameter('real', 1), Parameter('imag', 2)
        C = ComplexParameter('complex', A, B)
        return A, B, C

    @pytest.fixture
    def coupled(self):
        A = Parameter('base', 1)
        B = Parameter('modifier', 2, coupler=_coupler.AdditiveCoupler(A))
        C = ComplexParameter('complex', A, B)
        return A, B, C

    def test_is_parameter(self, uncoupled, coupled):
        _, _, C1 = uncoupled
        _, _, C2 = coupled
        assert isinstance(C1, Parameter)
        assert isinstance(C2, Parameter)

    def test_returns_complex(self, uncoupled, coupled):
        A1, B1, C1 = uncoupled
        A2, B2, C2 = coupled
        assert isinstance(C1.value, complex)
        assert isinstance(C2.value, complex)
        assert C1.value == A1.value + B1.value*1J
        assert C2.value == A2.value + B2.value*1J

    def test_get_value(self, coupled):
        A, B, C = coupled
        assert A.get_value() == 1
        assert B.get_value() == (1 + 2)
        assert C.get_value() == 1 + (1 + 2)*1J
        assert C.get_value(no_coupling=True) == 1 + 2J

    def test_setting_values_works(self, coupled):
        A, B, C = coupled
        assert C.value == 1 + (1+2)*1J
        B.set_value(1000)
        A.set_value(42)
        assert C.value == 42 + (1000+42)*1J
        with pytest.raises(TypeError):
            C.set_value(42 + 1042J)

    def test_bounds(self, uncoupled):
        _, _, C = uncoupled
        with pytest.raises(TypeError):
            C.bounds()


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

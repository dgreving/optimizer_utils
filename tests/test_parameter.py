import pytest
import logging
import sys

import xray_diffraction.datastructures.parameter as parameter
import xray_diffraction.datastructures.coupler as _coupler

from xray_diffraction.datastructures.parameter import IdentityCoupler
from xray_diffraction.datastructures.parameter import Parameter
from xray_diffraction.datastructures.parameter import ComplexParameter
from xray_diffraction.datastructures.parameter import ScatteringFactorParameter

from xray_diffraction.datastructures import parameter_exceptions as exc


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

    @pytest.mark.skip
    def test_get_and_set_coupler(self):
        pass

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


class TestComplexParameter:
    def test_is_parameter(self):
        real, imag = Parameter('first', 1), Parameter('second', 2)
        cp = ComplexParameter('complex', real, imag)
        assert isinstance(cp, Parameter)

    def test_is_complex(self):
        real, imag = Parameter('first', 1), Parameter('second', 2)
        cp = ComplexParameter('complex', real, imag)
        assert isinstance(cp.value, complex)

    def test_setting_values_works(self):
        real = Parameter('first', 1)
        imag = Parameter(
            'second',
            3,
            coupler=_coupler.AdditiveCoupler(base=real),
            )
        cp = ComplexParameter('complex', real, imag)
        assert cp.value == 1 + (1+3)*1J
        imag.set_value(1000)
        real.set_value(42)
        assert cp.value == 42 + (1000+42)*1J


class TestScatteringFactorParameter:

    @pytest.fixture(scope='function')
    def real_p(self):
        return Parameter('real', 1)

    @pytest.fixture(scope='function')
    def imag_p(self):
        return Parameter('imag', 2)

    @pytest.fixture(scope='function')
    def real_mag_p(self):
        return Parameter('real_mag', 3)

    @pytest.fixture(scope='function')
    def imag_mag_p(self):
        return Parameter('imag_mag', 4)

    @pytest.fixture(scope='function')
    def parameters(self):
        real, imag = Parameter('real', 1), Parameter('imag', 2)
        real_mag, imag_mag = Parameter('real_mag', 3), Parameter('imag_mag', 4)
        scatt_p = ScatteringFactorParameter(
            'scatt_fac_para', real, imag, real_mag, imag_mag)
        return real, imag, real_mag, imag_mag, scatt_p

    def test_is_parameter(self, real_p, imag_p, real_mag_p, imag_mag_p):
        scatt_p = ScatteringFactorParameter(
            'scatt_fac_para', real_p, imag_p, real_mag_p, imag_mag_p)
        assert isinstance(scatt_p, Parameter)

    def test_returns_complex_value(self, parameters):
        real, imag, real_mag, imag_mag, scatt_p = parameters
        assert isinstance(scatt_p.value, complex)
        assert scatt_p.value == (1 + 3) + (2 + 4)*1J

    def test_sensitive_to_para_modification(self, parameters):
        real, imag, real_mag, imag_mag, scatt_p = parameters
        real.set_value(100)
        assert scatt_p.value == (100 + 3) + (2 + 4)*1J

    def test_return_modes(self, parameters):
        scatt_p = parameters[-1]
        # test default behaviour
        assert scatt_p.return_mode == 'full'
        # test return_mode 'full'
        scatt_p.return_mode = 'full'
        assert scatt_p.value == (1 + 3) + (2 + 4)*1J
        # test return_mode charge, only non-magnetic contributions
        scatt_p.return_mode = 'charge'
        assert scatt_p.value == (1 + 0) + (2 + 0)*1J
        # test return_mode magn, only magnetic contributions considered
        scatt_p.return_mode = 'magn'
        assert scatt_p.value == (0 + 3) + (0 + 4)*1J
        # test return_mode '+', magnetism contributes positively
        scatt_p.return_mode = '+'
        assert scatt_p.value == (1 + 3) + (2 + 4)*1J
        # test return mode '-', magnetism contributes negatively
        scatt_p.return_mode = '-'
        assert scatt_p.value == (1 - 3) + (2 - 4)*1J


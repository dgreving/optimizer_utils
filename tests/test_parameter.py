import pytest
import logging
import sys

from xray_diffraction.datastructures.parameter import Parameter
from xray_diffraction.datastructures.parameter import IdentityCoupler


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

        coupler._couple(p)
        captured = capsys.readouterr()
        assert captured.out.startswith('Deprecation Warning!')

    def test_AdditiveCoupler(self):
        from xray_diffraction.datastructures.parameter import AdditiveCoupler
        base = Parameter(name='base', value=10)
        modifier = Parameter(name='modifier', value=5)
        coupler = AdditiveCoupler(base=base, modifier=modifier)
        assert coupler.value == 15

    def test_SubstractiveCoupler(self):
        from xray_diffraction.datastructures.parameter import SubtractiveCoupler
        base = Parameter(name='base', value=10)
        modifier = Parameter(name='modifier', value=5)
        coupler = SubtractiveCoupler(base=base, modifier=modifier)
        assert coupler.value == 5
        base = Parameter(name='base', value=10)
        coupled = Parameter(
            name='modifier',
            value=5,
            coupler=SubtractiveCoupler(base),
            )
        assert coupled.value == 5


class TestInterface:
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
        p.value = 150
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

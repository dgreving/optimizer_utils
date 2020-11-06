import logging
from abc import ABC, abstractmethod

from .compatability import Coupler_compatability_mixin
from .compatability import Parameter_Compatability_mixin

logger = logging.getLogger(__name__)


class Coupler(ABC, Coupler_compatability_mixin):
    def __init__(self, base=None, modifier=None, **kwargs):
        """
        Coupler class, which is initialised with an instance of a base
        parameter class, and then passed to another instance of Parameter as
        keyword argument 'coupler'.
        """
        self.base = base
        super().__init__(**kwargs)
        self.modifier = modifier

    @abstractmethod
    def value(self):
        raise NotImplementedError

    @abstractmethod
    def __repr__():
        raise NotImplementedError

    def couple(self, modifier):
        """Set modifier instance of Parameter class"""
        self.modifier = modifier


class IdentityCoupler(Coupler):
    """
    Default Coupler that does nothing but being present and wrapping the raw
    value of the underlying (not-)modifying Parameter
    """
    def coupling_func(self):
        msg = (
            'Deprecation Warning! '
            'Do not use method "coupling_func" on Coupler. '
            'Instead use property "value"')
        logger.warning(msg)
        return self.value

    @property
    def value(self):
        return self.modifier.get_value(no_coupling=True)

    def __repr__(self):
        return f'class IdentityCoupler, value: {self.modifier.value}'


class AdditiveCoupler(Coupler):
    def coupling_func(self):
        """calculate __add__ of two Parameter instances"""
        return self.value

    @property
    def value(self):
        return self.base + self.modifier

    def __repr__(self):
        return '{}(base: "{}") + {}(mod: "{}") = {}'.format(
            self.base.value,
            self.base.name,
            self.modifier.get_value(no_coupling=True),
            self.modifier.name,
            self.coupling_func(),
            )


class SubtractiveCoupler(Coupler):
    def coupling_func(self):
        return self.value

    @property
    def value(self):
        return self.base - self.modifier

    def __repr__(self):
        return '{}({}) - {}({}) = {}'.format(
            self.base.value,
            self.base.name,
            self.modifier.get_value(no_coupling=True),
            self.modifier.name,
            self.coupling_func(),
            )


class MultiplicativeCoupler(Coupler):
    def coupling_func(self):
        return self.base * self.modifier

    def __repr__(self):
        return '{}({}) * {}({}) = {}'.format(
            self.base.value,
            self.base.name,
            self.modifier.get_value(no_coupling=True),
            self.modifier.name,
            self.coupling_func(),
            )

# =============================================================================
# =============================================================================
# =============================================================================


class IParameter:
    def __init__(self, name, value=1., bounds=None, fit=False, coupler=None,
                 bounds_are_relative=False):
        self.name = name
        self._raw_val = value
        self.bounds_are_relative = bounds_are_relative
        self._bounds = bounds
        self.fit = fit
        self.coupler = coupler or IdentityCoupler(modifier=self)
        self.coupler.couple(self)

    def _textify(self):
        name = 'Name: "{}"'.format(self.name)
        value = '\tValue: {} ({})'.format(self._raw_val, self.value)
        bounds = '\tBounds: {}'.format(self.bounds)
        fit = '\tFit: {}'.format(self.fit)
        coupled_func = '\tCoupling: {}'.format(self.coupler)

        lst = [name, value, bounds, fit, coupled_func]
        text = ''.join(lst)
        return text

    def __str__(self):
        return self._textify()

    def __repr__(self):
        text = self._textify()
        return 'class: {}, {}'.format(self.__class__.__name__, text)


class Parameter(IParameter, Parameter_Compatability_mixin):
    """
    Basic building block of object oriented parameter treatment.

    Keyword arguments:
    name -- str
        Works as identifier
    value -- numerical
        Provides the raw value of the Parameter, which might be modifying an
        underlying base Parameter.
    bounds -- tuple
        Determines upper and lower bounds of fitting range. If
        bounds_are_relative=True each tuple element is treated as a factor
        upon 'value' attribute
    fit -- Bool
        Flag indicating whether parameter shall be fitted or not
    coupler -- instance of Coupler Class
        The Coupler must already be initialised and associated with a base
        Parameter. self is then treated as a modifier of base.
    bounds_are_relative -- Bool
        As explained under bounds
    """
    def __add__(self, other):
        return self.coupler.coupling_func() + other._raw_val

    def __sub__(self, other):
        return self.coupler.coupling_func() - other._raw_val

    def __mul__(self, other):
        return self.coupler.coupling_func() * other._raw_val

    def __call__(self):
        return self.value

    @property
    def bounds(self):
        if not self.bounds_are_relative:
            return self._bounds
        else:
            return (self._bounds[0] * self.value, self._bounds[1] * self.value)

    @bounds.setter
    def bounds(self, new_bounds):
        self.bounds = new_bounds

    @property
    def raw_bounds(self):
        return self.bounds

    def get_value(self, no_coupling=False):
        if no_coupling:
            return self._raw_val
        else:
            return self.coupler.coupling_func()

    def set_value(self, value):
        self._raw_val = value

    @property
    def value(self):
        return self.coupler.value

    @value.setter
    def value(self, attr):
        self._raw_val = attr

    def update(self, value):
        self._raw_val = value

    def to_contr(self, controller):
        controller.add_parameters(self)
        return self

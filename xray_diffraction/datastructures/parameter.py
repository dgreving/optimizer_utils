import logging
from abc import ABC, abstractmethod

from .compatability import Parameter_Compatability_mixin

from .coupler import IdentityCoupler

from . import parameter_exceptions as exc

logger = logging.getLogger(__name__)


# =============================================================================
# =============================================================================
# =============================================================================


class IParameter:
    def __init__(self, name, value=None, bounds=None, fit=False, coupler=None,
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
    def __call__(self):
        return self.value

    def set_value(self, value):
        self._raw_val = value

    @property
    def value(self):
        logger.debug('calling Parameter.value')
        return self.coupler.value

    @value.setter
    def value(self, attr):
        self._raw_val = attr

    def get_value(self, no_coupling=False):
        if no_coupling:
            return self._raw_val
        else:
            return self.coupler.coupling_func()

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

    def update(self, value):
        self._raw_val = value

    def to_contr(self, controller):
        controller.add_parameters(self)
        return self


class ComplexParameter(Parameter):

    def __init__(self, name, real_part, imag_part=None):
        super().__init__(name)
        self.real = real_part
        self.imag = imag_part or Parameter(name='imag', value=0.)

    @property
    def value(self):
        return self.real.value + 1J * self.imag.value

    def set_value(self, complex_value):
        self.real.value = complex_value.real
        self.imag.value = complex_value.imag


class ScatteringFactorParameter(Parameter):
    """
    Construct of 2 to 4 Parameter instances representing real- and imaginary-,
    charge- and magnetic- parts of a scattering element.

    Keyword argument:
    name -- str
        identifier
    f_charge_real -- instance of Parameter class
        Real part of the charge component of the scattering factor
    f_charge_imag -- instance of Parameter class
        Imaginary part of the charge component of the scattering factor
    f_magn_real -- instance of Parameter class
        Real part of magnetic contribution to scattering factor
    f_magn_imag -- instance of Parameter class
        Imaginary part of magnetic contribution to scattering factor
    return_mode -- str, one of 'full', 'charge', 'magn', '+', '-'
        Indicates return mode of the scattering factor. Might be only charge,
        only magnetic, or adding or subtracting the magnetic contribution,
    """
    def __init__(self, name,
                 f_charge_real, f_charge_imag,
                 f_magn_real=None, f_magn_imag=None,
                 return_mode='full'):
        super().__init__(name)
        self.f_ch_r = f_charge_real
        self.f_ch_i = f_charge_imag
        self.f_m_r = f_magn_real or Parameter('f_mag', 0.)
        self.f_m_i = f_magn_imag or Parameter('f_mag', 0.)

        self.return_mode = return_mode

    def set_return_mode(self, return_mode):
        self.return_mode = return_mode

    @property
    def value(self):
        logger.debug(f'Calling ScatteringFactorPara: mode={self.return_mode}')
        if self.return_mode == 'full':
            return self.f_ch_r()+self.f_m_r() + 1j*(self.f_ch_i()+self.f_m_i())
        elif self.return_mode in ['charge', 'c']:
            return self.f_ch_r() + 1j * self.f_ch_i()
        elif self.return_mode in ['magn', 'mag', 'magnetic', 'm']:
            return self.f_m_r() + 1j * self.f_m_i()
        elif self.return_mode in ['+', 'plus']:
            return self.f_ch_r()+self.f_m_r() + 1j*(self.f_ch_i()+self.f_m_i())
        elif self.return_mode in ['-', 'minus']:
            return self.f_ch_r()-self.f_m_r() + 1j*(self.f_ch_i()-self.f_m_i())
        else:
            raise NameError('ScatteringFactorParameter return mode unknown.')

    def get_value(self):
        return self.value

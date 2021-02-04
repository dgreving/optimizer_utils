import logging
from abc import ABC, abstractmethod

from .coupler import NoCoupler, IdentityCoupler, coupler_map
from .coupler import AdditiveCoupler, SubtractiveCoupler, MultiplicativeCoupler

logger = logging.getLogger(__name__)


# =============================================================================
# =============================================================================
# =============================================================================


class IParameter(ABC):
    """Interface class to parameter objects"""
    @property
    @abstractmethod
    def value(self):
        raise NotImplementedError

    @abstractmethod
    def set_value(self, value):
        raise NotImplementedError

    @abstractmethod
    def get_value(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def bounds(self):
        raise NotImplementedError

    def to_contr(self, controller):
        controller.add_parameters(self)
        return self

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


class Parameter(IParameter):
    """
    Basic building block of object oriented parameter treatment.

    Keyword arguments:

    name -- str
        Works as identifier
    value -- float
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

    def __init__(self, name, value=None, bounds=None, fit=False, coupler=None,
                 bounds_are_relative=False):
        self.name = name
        self._raw_val = value
        self._bounds = bounds
        self.bounds_are_relative = bounds_are_relative
        if type(coupler) == tuple:
            identifier, base = coupler
            _coupler = coupler_map[identifier]
            self.coupler = _coupler(base)
        else:
            self.coupler = coupler or NoCoupler(modifier=self)
        self.coupler.couple(self)
        self._fit = None
        self.fit = fit

    @property
    def bounds(self):
        if self._bounds is None:
            return self._bounds
        if not self.bounds_are_relative:
            return self._bounds
        else:
            return self._bounds[0] * self.value, self._bounds[1] * self.value

    @bounds.setter
    def bounds(self, new_bounds):
        self._bounds = new_bounds

    @property
    def fit(self):
        return self._fit

    @fit.setter
    def fit(self, value):
        if (value is False) or (value is None):
            self._fit = value
        else:
            if self.bounds is not None:
                self._fit = value
            else:
                msg = (
                    f'Toggling fit=True only allowed on set bounds.',
                    f'(Parameter: {self.name})'
                    )
                raise AttributeError(msg)

    @property
    def value(self):
        logger.debug('calling Parameter.value')
        return self.coupler.value

    def set_value(self, value):
        """
        Sets value of parameter.
        If parameter is coupled, sets only the raw value, i.e. modifier value
        NOT the coupled value as obtained from calling Parameter.get_value()
        """
        self._raw_val = value

    def get_value(self, no_coupling=False):
        """
        Return parameter value
        If parameter is coupled and no_coupling == True, return uncoupled value
        """
        if no_coupling:
            return self._raw_val
        else:
            return self.value


# ==============================================================================
# ==============================================================================
# ==============================================================================


class ReferenceParameter(IParameter):
    """
    Is identical to a referenced parameter apart from its name
    """
    def __init__(self, name, references):
        """
        Creates parameter identical to reference apart form its name
        :param name: Identifier of this parameter
        :param references: Parameter that all other properties are shared with
        """
        self.name = name
        self.coupler = IdentityCoupler(references)
        self.coupler.couple(self)
        self._raw_val = None

    @property
    def value(self):
        return self.coupler.value

    def set_value(self, value):
        msg = (
            'Can not assign value to ReferenceParameter\n'
            'Assign value to referenced parameter instead.'
        )
        raise TypeError(msg)

    def get_value(self, no_coupling=False):
        if no_coupling:
            return self.coupler.base._raw_val
        else:
            return self.coupler.base.value

    @property
    def bounds(self):
        return self.coupler.base.bounds

    @bounds.setter
    def bounds(self, bounds):
        msg = (
            'Can not assign to bounds attribute of ReferenceParameter.\n'
            'Assign to "bounds" of referenced Parameter instead.'
        )
        raise TypeError(msg)

    @property
    def fit(self):
        return None

    @fit.setter
    def fit(self, value):
        msg = (
            'Can not assign to "fit" attribute of ReferenceParameter.\n'
            'Assign to "fit" of referenced Parameter instead.'
        )
        raise TypeError(msg)


# ==============================================================================
# ==============================================================================
# ==============================================================================


class ComplexParameter(Parameter):
    """
    Composite object of two Parameter instances, each representing real and
    imaginary part of a complex number.
    All interaction should be limited to real- and imag-attributes.
    Returns a complex number, given by the values of real- and imaginary
    components
    """

    def __init__(self, name, real_part, imag_part=None):
        """
        :param name: Identifier of this parameter
        :param real_part: Parameter representing real part of complex number
        :param imag_part: Parameter representing imaginary part of complex number
        """
        super().__init__(name)
        self.real = real_part
        self.imag = imag_part or Parameter(name='imag', value=0.)

    @property
    def value(self):
        return self.real.value + 1J * self.imag.value

    def set_value(self, value):
        raise TypeError(
            'Use set_value() method of attributes "real" and "imag" '
            'instead of ComplexParameter.')

    def get_value(self, no_coupling=False):
        if no_coupling:
            return self.real._raw_val + 1J * self.imag._raw_val
        else:
            return self.value

    @property
    def bounds(self):
        raise TypeError(
            'Use "bounds" attribute of "real" and "imag" attributes of '
            'ComplexParameter instance.'
            )


# ==============================================================================
# ==============================================================================
# ==============================================================================


class ScatteringFactorParameter(Parameter):
    """
    Construct of 2 to 4 Parameter instances representing real- and imaginary-,
    charge- and magnetic- parts of a scattering element.
    """
    def __init__(self, name,
                 f_charge_real, f_charge_imag,
                 f_magn_real=None, f_magn_imag=None,
                 return_mode='full'):
        """
        :param name: Identifier of this Parameter
        :param f_charge_real: instance of Parameter class
            Imaginary part of the charge component of the scattering factor
        :param f_charge_imag: instance of Parameter class
            Imaginary part of the charge component of the scattering factor
        :param f_magn_real: f_magn_real -- instance of Parameter class
            Real part of magnetic contribution to scattering factor
        :param f_magn_imag: instance of Parameter class
        Imaginary part of magnetic contribution to scattering factor
        :param return_mode: str, one of 'full', 'charge', 'magn', '+', '-'
            Indicates return mode of the scattering factor. Might be only charge,
            only magnetic, or adding or subtracting the magnetic contribution,
        """
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
            return self.f_ch_r.value + self.f_m_r.value \
                + 1J * (self.f_ch_i.value + self.f_m_i.value)

        elif self.return_mode in ['charge', 'c']:
            return self.f_ch_r.value + 1J * self.f_ch_i.value

        elif self.return_mode in ['magn', 'mag', 'magnetic', 'm']:
            return self.f_m_r.value + 1J * self.f_m_i.value

        elif self.return_mode in ['+', 'plus']:
            return (self.f_ch_r.value + self.f_m_r.value) \
                + (self.f_ch_i.value + self.f_m_i.value) * 1J

        elif self.return_mode in ['-', 'minus']:
            return (self.f_ch_r.value - self.f_m_r.value) \
                + (self.f_ch_i.value - self.f_m_i.value) * 1J

        else:
            raise NameError('ScatteringFactorParameter return mode unknown.')

    def get_value(self):
        return self.value


# ==============================================================================
# ==============================================================================
# ==============================================================================


class ParameterGroup(Parameter):
    """
    Turns a set of ordered parameters into an iterable (list)

    Upon calling "value", return a list of the underlying parameter-values
    """
    def __init__(self, group_name, *parameters):
        """
        :param group_name: Identifier of this Parameter group
        :param parameters: Parameters to join the group in the given order
        """
        self.name = group_name
        self.group = [p for p in parameters]
        self.fit = None

    @property
    def value(self):
        """
        :return: List of values of the individual parameters of the group
        """
        return [p.value for p in self.group]

    def set_value(self, value):
        raise TypeError(
            'Use set_value() method of constituent parameters '
            'instead of ParameterGroup instance.')

    def get_value(self, no_coupling=False):
        if no_coupling:
            return [p._raw_val for p in self.group]
        else:
            return self.value

    @property
    def bounds(self):
        raise TypeError(
            'Use "bounds" attribute of constituent parameters '
            'instead of ParameterGroup instance.'
            )

    def __repr__(self):
        name = 'Name: {}(ParameterGroup)'.format(self.name)
        return " ".join([name] + [str(p.value) for p in self.group])

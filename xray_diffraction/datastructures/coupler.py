from abc import ABC, abstractmethod
import logging


logger = logging.getLogger(__name__)


class Coupler(ABC):
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


class ArithmeticCoupler(Coupler):
    my_dict = {
        '+': lambda x, y: x + y,
        '-': lambda x, y: x - y,
        '*': lambda x, y: x * y,
    }
    _op = None

    @property
    def value(self):
        logger.debug('calling AdditiveCoupler.value')
        return self.my_dict[self._op](self.base.value, self.modifier._raw_val)

    def __repr__(self):
        return '{}(base: "{}") {} {}(modifier: "{}") = {}'.format(
            self.base.value,
            self.base.name,
            self._op,
            self.modifier.get_value(no_coupling=True),
            self.modifier.name,
            self.value,
            )


class AdditiveCoupler(ArithmeticCoupler):
    _op = '+'


class SubtractiveCoupler(ArithmeticCoupler):
    _op = '-'


class MultiplicativeCoupler(ArithmeticCoupler):
    _op = '*'


coupler_map = {
    'additive': AdditiveCoupler,
    'subtractive': SubtractiveCoupler,
    'multiplicative': MultiplicativeCoupler,
}

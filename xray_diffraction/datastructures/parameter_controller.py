import pickle
import contextlib
import numpy as np
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class ParameterController:
    """
    Containerclass for collections of Parameter instances.
    """
    def __init__(self, name='', suffix=''):
        self._mapping = OrderedDict()
        self.name = name
        self.suffix = suffix

    def add(self, parameter, *parameters):
        for p in (parameter,) + parameters:
            suffixed = self._suffixed_name(p.name)
            unsuffixed = self._unsuffixed_name(p.name)
            if suffixed in self._mapping and p is not self.get(unsuffixed):
                raise ValueError(
                    f'Different parameter of name "{unsuffixed}" exists already'
                )
            p.name = suffixed
            self._mapping[suffixed] = p

    def get(self, key):
        """
        Returns a parameter of a certain identifier.
        Note that the identifier does NOT contain the controller suffix.
        """
        try:
            return self._mapping[self._suffixed_name(key)]
        except KeyError:
            raise KeyError(
                f'No parameter of name "{key}" in controller "{self.name}".'
            )

    def get_value(self, key, no_coupling=False):
        return self.get(key).get_value(no_coupling=no_coupling)

    def __getitem__(self, key):
        """
        Fetches parameter instance by name

        :param key: Name attribute of the parameter
        :return: parameter instance with attribute name==key
        """
        return self.get(key)

    def __iter__(self):
        for name in self._mapping:
            yield self._unsuffixed_name(name)

    def update(self, *key_val_tuples, **kwargs):
        """
        Updates parameter values identified by name.

        Keyword arguments:

        *key_val_tuples -- tuples
            Tuples of form (parameter.name, new_value)
        **kwargs -- key -> parameter.name, value -> new_value

        :return: None
        """
        try:
            for key, val in key_val_tuples:
                self.get(key).set_value(val)
            for key, val in kwargs.items():
                self.get(key).set_value(val)
        except KeyError:
            raise KeyError(f'Parameter not in controller: {key}')

    def _suffixed_name(self, name):
        name, _, suffix = name.partition('__')
        if self.suffix:
            return name + f'__{self.suffix}'
        else:
            return name

    def _unsuffixed_name(self, name):
        name, _, suffix = name.partition('__')
        return name

    def merge(self, other):
        """
        Merges in an external ParameterController, by adding all parameter objects

        :param other: Iterable mapping, names -> parameter  instances, e.g.
            ParameterController instance
        :return: Calling instance
        """
        for key in other:
            self.add(other.get(key))
        return self

    def values(self, only_fitted=False):
        for para in self._mapping.values():
            if not only_fitted:
                yield para
            elif para.fit:
                yield para

    def _keys(self, only_fitted=False):
        for para in self.values(only_fitted=only_fitted):
            yield para

    def keys(self, only_fitted=False, suffixed=True):
        for para in self._keys(only_fitted=only_fitted):
            if suffixed:
                yield self._suffixed_name(para.name)
            else:
                yield self._unsuffixed_name(para.name)

    def bounds(self, only_fitted=False):
        for para in self.values(only_fitted=only_fitted):
            yield para.bounds

    def as_list(self, only_fitted=False):
        return list(self.values(only_fitted=only_fitted))

    def num_paras(self, only_fitted=False):
        return len(self.as_list(only_fitted=only_fitted))

    def __repr__(self):
        separator = '---------------------------------\n'
        header = '\n\nClass: {}\n'.format(self.__class__.__name__) + 2*separator
        body = separator.join([str(p) + '\n' for p in self.values()])
        return header + body + separator


import pickle
import contextlib
import numpy as np
from collections import OrderedDict


class ParameterController():
    """
    Containerclass for collections of Parameter instances.
    """
    def __init__(self, name='', para_suffix='', collection=None):
        self._od = OrderedDict()
        self.name = name
        self.para_suffix = para_suffix
        self.collection = collection

    def add_parameter(self, parameter):
        self._od[parameter.name] = parameter

    def get_parameter(self, key):
        return self._od[key]

    def update(self):
        pass

    def _suffixed_name(self, parameter):
        name, suffix = parameter.name.split('__')
        pass

    def _unsuffixed_name(self, parameter):
        pass


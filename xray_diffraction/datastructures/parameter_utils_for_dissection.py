from collections import OrderedDict
import numpy as np
import pickle
from abc import ABC, abstractmethod
import contextlib

import logging

logger = logging.getLogger(__name__)











# =============================================================================
# =============================================================================
# =============================================================================


class ControllerCollection(OrderedDict):
    """
    Wrapper class of OrderedDict, expecting instances of ParameterController,
    providing easy pickling and remote access from ParameterController
    instances.

    When added to a ControllerCollection, the save method a ParameterController
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._notify_controllers()
        self.parameter_list = None

    def _notify_controllers(self):
        for controller in self.values():
            if hasattr(controller, 'add_to_collection'):
                controller.add_to_collection(self)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            collection = pickle.load(f)
        return collection

    def create_parameter_list(self):
        lst = []
        for controller in self.values():
            for p in controller.plain_paras_to_list():
                if p not in lst:
                    lst.append(p)
        self.parameter_list = lst

    def __repr__(self):
        controllers_str = ', '.join(self.keys())
        return f"ControllerCollection containing: <{controllers_str}>"


# =============================================================================
# =============================================================================
# =============================================================================


def set_mode(controller, keys, mode):
    for key in keys:
        controller.get_parameter(key).set_return_mode(mode)


# =============================================================================
# =============================================================================
# =============================================================================

def load_controller(filename):
    with open(filename, 'rb') as f:
        controller = pickle.load(f)
    return controller


def init_population(popsize, controller, scale):
    """
    Returns an initialiser for differential evolution.
    All parameters are initilised at random, apart from the ones defined in
    'scale' parameter, which are initialised with a normal distribution
    around a specific value.

    Keyword arguments:
    popsize -- int
        population size of differential evolution algorithm
    controller -- controller instance
    scale -- iterable containing tuples of 'fit_key', 'loc' and 'scale'
    """
    fit_keys, bounds = controller.extract_keys_and_bounds()
    num_par = len(fit_keys)
    init_arr = np.empty((num_par * popsize, num_par))
    scale_keys = [s[0] for s in scale]
    i = 0
    for fit_key, bound in zip(fit_keys, bounds):
        if fit_key not in scale_keys:
            init_arr[:, i] = np.random.uniform(
                bound[0],
                bound[1],
                num_par*popsize,
                )
        else:
            print('doing it as index: ', i)
            scale_index = scale_keys.index(fit_key)
            print('scale index: ', scale_index)
            print(scale[scale_index][1], scale[scale_index][2])
            distr = np.random.normal(
                loc=scale[scale_index][1],
                scale=scale[scale_index][2],
                size=num_par*popsize,
                )
            init_arr[:, i] = distr
        i += 1
    return init_arr

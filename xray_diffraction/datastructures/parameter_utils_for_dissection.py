from collections import OrderedDict
import numpy as np
import pickle
from abc import ABC, abstractmethod
import contextlib

import logging

logger = logging.getLogger(__name__)








class ParameterControllerExtension:
    def create_parameters(self, *paradicts):
        parameters = [Parameter(**paradict) for paradict in paradicts]
        if self.para_suffix:
            for p in parameters:
                p.name = '_'.join([p.name, self.para_suffix])
        self.add_parameters(*parameters)
        return self

    def _get_sfp_part(self, para_name, alt_name):
        try:
            if para_name is not None:
                return self[para_name]
            else:
                return self[alt_name]
        except KeyError:
            raise KeyError('Every scattering factor part has to be defined')

    def _suffixed_name(self, name):
        if self.para_suffix:
            return '_'.join([name, self.para_suffix])
        else:
            return name

    def create_scattering_factor_parameter(
            self, name, ch_real=None, ch_imag=None, mag_real=None,
            mag_imag=None):
        """
        Creates and adds ScatteringFactorParameter to ParameterController.

        Keyword arguments:
        name -- str
            Name of the Scattering Factor Parameter to be created
        ch_real -- str
            Name of the Parameter ALREADY PRESENT in the controller, which
            represents the real charge part. If ch_real is not set, there HAS
            to be a parameter named 'name_ch_real' present, which might simply
            be an empty parameter Parameter(name='undefined', value=0.) etc.
        other attributes -- str
            Same as above, just considering the other parts of the sfp
        """

        part_names = ch_real, ch_imag, mag_real, mag_imag
        identifiers = 'ch_real', 'ch_imag', 'mag_real', 'mag_imag'
        alt_names = ['_'.join([name, x]) for x in identifiers]
        f_charge_real, f_charge_imag, f_magn_real, f_magn_imag = [
            self._get_sfp_part(name, alt_name)
            for name, alt_name in zip(part_names, alt_names)
            ]

        suffixed_name = self._suffixed_name(name)
        self[name] = ScatteringFactorParameter(
            suffixed_name, f_charge_real, f_charge_imag, f_magn_real,
            f_magn_imag, return_mode='c')

    def create_parameter_group(self, group_name, *member_names):
        member_paras = [self[name] for name in member_names]
        group = ParameterGroup(self._suffixed_name(group_name), *member_paras)
        self.add_parameters(group)


class ParameterController(OrderedDict, ParameterControllerExtension):
    """
    help
    """
    def __init__(self, name='', para_suffix=''):
        self.collection = None
        self._direct_values = False
        self.name = name
        self.para_suffix = para_suffix

    def set_name(self, name):
        self.name = name
        return self

    def add_to_collection(self, collection):
        self.collection = collection

    def _internal_name(self, parameter):
        if parameter.name.endswith('_'+self.para_suffix):
            striplen = len('_'+self.para_suffix)
            return parameter.name[:0-striplen]
        else:
            return parameter.name

    def _add_parameter(self, parameter, key=None):
        """
        Add parameter only if not yet present in controller
        """
        if key is None:
            internal_name = self._internal_name(parameter)
        else:
            internal_name = key
        if internal_name not in self:
            self[internal_name] = parameter
        else:
            msg = 'Parameter {} already exists'.format(parameter.name)
            raise ValueError(msg)

    def _add_parameter_no_exception(self, parameter):
        """
        Add parameter if not yet present in controller, do nothing if parameter
        is already present and raise an exception if attempting to add
        different parameter of already present name.
        """
        is_present = self._internal_name(parameter) in self

        # add parameter if not yet present
        if not is_present:
            self._add_parameter(parameter)
            # self[parameter.name] = parameter
        # raise Exception on attempt to add non-identical para of the same name
        elif is_present \
                and (parameter is not self[self._internal_name(parameter)]):
            msg = """
            Parameters of same name "{}" in merged controller are
            not the same objects.
            """.format(parameter.name)
            raise ValueError(msg)
        # if the exact parameter object is already present do nothing
        elif is_present \
                and (parameter is self[self._internal_name(parameter)]):
            pass

    def add_parameters(self, *parameters):
        if not parameters:
            return self
        for p in parameters:
            if isinstance(p, tuple):
                self._add_parameter(p[0], p[1])  # parameter at 0, key at 1
            else:
                self._add_parameter(p)

        return self

    def update_parameters(self, *key_val_pairs):
        for key, val in key_val_pairs:
            if key in self:
                self[key].update(val)
            else:
                msg = """Parameter {} has not been previously present.
                    Consider spellchecking.""".format(key)
                raise ValueError(msg)

    @property
    def num_fitted_parameters(self):
        return len(self.return_parameter_list(only_fitted=True))

    def return_parameter_list(self, only_fitted=False):
        if not only_fitted:
            return [p for p in self.values()]
        else:
            return [p for p in self.values() if p.fit]

    def get_value(self, key):
        # fetches parameter and returns parameter.value
        return self[key].value

    def __getitem__(self, key):
        try:
            if not self._direct_values:
                return super().__getitem__(key)
            elif self._direct_values:
                return super().__getitem__(key).value
        except KeyError:
            raise KeyError(f'Unknown key to ParameterController: "{key}"')

    def get_parameter(self, key):
        # identical with __getitem__
        return self.__getitem__(key)

    def merge_in_controller(self, *controller):
        """
        Create the unity of multiple controllers, i.e. add each parameter
        present in one or more controllers exactly once to this controller,
        while ensuring that no different parameter objects of the same name can
        be merged.

        Returns self after merge has been performed
        """
        for c in controller:
            for key in c:
                self._add_parameter_no_exception(c.get_parameter(key))
        return self

    def extract_keys_and_bounds(self, only_fitted=True):
        p_lst = self.return_parameter_list(only_fitted)
        fit_keys = [p.name for p in p_lst]
        fit_bounds = [p.bounds for p in p_lst]
        return fit_keys, fit_bounds

    def extract_vals(self, only_fitted=True):
        p_lst = self.return_parameter_list(only_fitted)
        return [p.get_value(no_coupling=True) for p in p_lst]

    @property
    def fit_keys(self):
        return [key for key in self if self[key].fit]

    def toggle_all_fits_inactive(self):
        for name, parameter in self.items():
            if hasattr(parameter, 'fit'):
                parameter.fit = False

    def to_dict(self):
        # Turn self into simple dict, losing all extra functionality
        return {name: parameter.value for name, parameter in self.items()}

    def plain_paras_to_dict(self):
        # Return ONLY objects of plain Parameter type, i.e. no subclasses etc.
        return {k: self[k] for k in self if isinstance(self[k], Parameter)}

    def plain_paras_to_list(self):
        return [p for p in self.values() if type(p) == Parameter]

    def _random(self, lower, upper):
        return np.random.uniform(lower, upper)

    def randomise_within_bounds(self, seed=None, randomise_all=False):
        """
        Generally randomises fitted parameters, unless randomise_all == True,
        which fits all parameters that obtain valid bounds.
        Optional seed pkeyword argument ensures reproducible randomisation,
        which is reverted once the function returns, i.e. seeded state is
        reverted to previous state after function call.
        """
        state = np.random.get_state()
        if seed:
            np.random.seed(seed)
        for name, para in self.items():
            if para.fit or (randomise_all and para.bounds):
                rnd_val = self._random(para.bounds[0], para.bounds[1])
                self.update_parameters((para.name, rnd_val))
        np.random.set_state(state)
        return self

    def __repr__(self):
        separator = '\n---------------------------------\n'
        header = 'Class: {}'.format(self.__class__.__name__) + separator
        body = separator.join([str(p) for p in self.values()])
        return header + body + separator

    def set_mode(self, keys, mode, set_all_available):
        """
        Sets mode of specified keys

        If 'set_all_available' flag is set, the mode of every parameter that
        has a "set_return_mode" method is set to mode.
        """
        if set_all_available:
            #
            for p in self.values():
                if hasattr(p, 'set_return_mode'):
                    p.set_return_mode(mode)
            #
            # for key in self:
            #     atr = getattr(self.get_parameter(key), 'set_return_mode', None)
            #     if atr:
            #         atr(mode)
        else:
            for key in keys:
                self.get_parameter(key).set_return_mode(mode)

    def update_from_controller(self, controller):
        for key in self:
            has_attr = hasattr(self[key], 'set_value')
            is_no_scattfac = not isinstance(
                                          self[key], ScatteringFactorParameter)
            if key in controller and has_attr and is_no_scattfac:
                self[key].value = controller[key].get_value(no_coupling=True)

    def _save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def save(self, filename, save_collection=True):
        if not save_collection:
            self._save(filename)
        elif save_collection:
            self.collection.save(filename)

    @contextlib.contextmanager
    def direct_values(self):
        self._direct_values = True
        try:
            yield
        finally:
            self._direct_values = False


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

#!/usr/bin/env python3
"""
Demonstrate general use of simultaneous fitting of multiple (individually
under-determined, i.e. coupled implicitly amongst,) datasets
"""
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

from xray_diffraction import ParameterController, Fitter, Dataset
from xray_diffraction import parameter


def gaussian_error(data, scale):
    return data + np.random.normal(scale=scale, size=len(data))


class DataClass(ABC):
    def __init__(self, x, parameter_controller):
        self.x = x
        self.parameter_controller = parameter_controller

    @abstractmethod
    def eval(self):
        """
        Simulation function, generating the signal that is used
        for data optimisation.
        Will be passed to Dataset instance and has to return x, y tuple
        in order for the figure of merit to be calculated correctly.
        :return: x, y (array_like, abscissa and ordinate function values)
        """
        raise NotImplementedError


class LinearData(DataClass):
    """
    Provides necessary functionality for modelling a linear relationship
    of data, depending on the parameter values collected in
    parameter_controller.

    Note that the slope being a combination of two parameters does not allow to
    uniquely resolve the parameters on individual fitting of this dataset.
    """
    def eval(self):
        component_1 = self.parameter_controller.get_value('p1')
        component_2 = self.parameter_controller.get_value('p2')
        return self.x, (component_1 + component_2) * self.x


class SinusoidalData(DataClass):
    """
    Provides necessary functionality for modelling a sinusoidal relationship
    of data, depending on the parameter values collected in
    parameter_controller.

    Note that the wavelength (lambda) being a combination of two parameters does
    not allow to uniquely resolve the parameters on individual fitting of this
    dataset.
    """
    def eval(self):
        lambda_1 = self.parameter_controller.get_value('p1')
        lambda_2 = self.parameter_controller.get_value('p2')
        lambda_ = lambda_1 - lambda_2
        amp = self.parameter_controller.get_value('same_as_p3')
        offset = self.parameter_controller.get_value('p4')
        return self.x, amp * np.sin(2 * np.pi * self.x / lambda_) + offset


parameter_controller = ParameterController()

parameter_controller.add(
    parameter.Parameter(
        'p1',
        value=4,
        bounds=(2, 10),
        fit=True,
        ),
    parameter.Parameter(
        'p2',
        value=2,
        bounds=(0.1, 10),
        bounds_are_relative=True,
        fit=True,
        ),
    parameter.Parameter(
        'p3',
        value=15,
        bounds=(5, 32),
        fit=True,
        ),
    parameter.Parameter(
        'p4',
        value=5,
        fit=False,
    )
)

ref = parameter.ReferenceParameter('same_as_p3', parameter_controller['p3'])
parameter_controller.add(ref)


# used for final comparison
true_values = [parameter_controller.get_value(k) for k in parameter_controller]


def create_dataset(parameter_controller, x, constructor, err_std_abs):
    """
    Create instances of the Dataset class that are holding (fake) experimental
    data and produce simulated data used for fitting.

    :param parameter_controller: Holds all parameter objects referenced by the
        simulation function
    :param x: array-like, the abscissa values of the data
    :param constructor: constructor class used to define the simulation function
    :param err_std_abs: absolute value of the error standard deviation, used to
        artificially create noisy (fake-experimental) data
    :return: x_values, true_y, noisy_y, dataset
    """
    data = constructor(x, parameter_controller)
    true_x, true_y = data.eval()
    noisy_y = gaussian_error(data=true_y, scale=err_std_abs)
    dataset = Dataset(true_x, noisy_y, sim_func=data.eval)
    return x, true_y, noisy_y, dataset


x_1, true_y_1, noisy_y_1, dataset_1 = create_dataset(
    parameter_controller=parameter_controller,
    x=np.linspace(0, 10, 51),
    constructor=LinearData,
    err_std_abs=3,
    )


x_2, true_y_2, noisy_y_2, dataset_2 = create_dataset(
    parameter_controller=parameter_controller,
    x=np.linspace(-5, 5, 51),
    constructor=SinusoidalData,
    err_std_abs=3,
    )


# add an optional (small) artifact to visualize effect of model imperfection
if True:
    true_y_1 = true_y_1 + 20*np.exp(-(x_1 - 5)**2 / 0.3)
    noisy_y_1 = noisy_y_1 + 20*np.exp(-(x_1 - 5)**2 / 0.3)
    dataset_1.y = noisy_y_1


fitter = Fitter(master_controller=parameter_controller)
fitter.add_dataset(dataset_1, fit=True, fom_type='diff')
fitter.add_dataset(dataset_2, fit=True,  fom_type='diff')

result = fitter.optimize()


msg = f''
msg += f'fit: {result.success}\n\n'
msg += '\n'.join(
    [
        f'{fit} -> {true}'
        for fit, true in zip(true_values, result.x)
        ]
    )
print(msg)


xs_1, ys_1 = dataset_1.simulate()
xs_2, ys_2 = dataset_2.simulate()
fom_array_1, fom_array_2 = fitter.fom_handler.fom_arrays


def plot_data(x, true_y, noisy_y, sim_y, fom_array):
    fig, axes= plt.subplots(
        ncols=1, nrows=2, sharex=True,
        gridspec_kw={'hspace': 0, 'height_ratios': [2, 1]},
        )
    ax1, ax2 = axes
    ax1.scatter(x, noisy_y, marker='x', c='k')
    ax1.plot(x, true_y, color='k', linestyle='--')
    ax1.plot(x, sim_y, color='g')
    ax2.plot(x, fom_array, color='r')

    ax2.set_xlabel('x')
    ax1.set_ylabel('y')
    ax2.set_ylabel('figure of merit')


plot_data(x_1, true_y_1, noisy_y_1, ys_1, fom_array_1)
plot_data(x_2, true_y_2, noisy_y_2, ys_2, fom_array_2)


plt.show()

import numpy as np
import matplotlib.pyplot as plt

from xray_diffraction import ParameterController, Fitter, Dataset
from xray_diffraction import parameter


def poisson_error(y):
    return y + np.random.normal(scale=np.sqrt(y))


def make_sublayer(z, z0, profile, t, dens):
    return np.where(np.logical_and(z > z0, z <= z0+t), dens, profile)


def make_layer(z, z0, profile, thicknesses, densities):
    for t, dens in zip(thicknesses, densities):
        profile = make_sublayer(z, z0, profile, t, dens)
        z0 += t
    return profile


def make_stack(z, thicknesses, densities, num_rep):
    total_layer_thickness = np.sum(thicknesses)
    profile = np.zeros_like(z)
    for i in range(num_rep):
        profile = make_layer(
            z, i*total_layer_thickness, profile, thicknesses, densities)
    return profile


class ReflectivityData:
    """
    Container class producing electron density profiles and corresponding
    (massively) simplified x-ray reflectivity by calculating the Fast Fourier
    Transformation of the density profile, depending on the current state of
    the parameters collected within the parameter_controller.
    """
    def __init__(self, parameter_controller):
        self.dz = 0.01
        self.dQz = 0.00001
        self.Qz_max = 0.25
        self.parameter_controller = parameter_controller

    def create_electron_density_profile(self):
        """
        Create an arbitrary multilayer stack of a repeating "unit-cell-layer"
        consisting of an arbitrary number of sublayers.

        :return: numpy-arrays -> z-values, electron_density
        """
        layer_thicknesses = self.parameter_controller.get_value('thicknesses')
        densities = self.parameter_controller.get_value('densities')
        pc = self.parameter_controller
        z_max = pc.get_value('layer_reps') * np.sum(pc.get_value('thicknesses'))
        z_array = np.arange(0, z_max + 10, self.dz)
        profile = make_stack(
            z_array,
            layer_thicknesses,
            densities,
            num_rep=pc.get_value('layer_reps'),
        )
        return z_array, profile

    def fft_profile(self):
        """
        Calculate the FFT of the electron density profile, representing a
        simplified way of approximating x-ray reflectivity of a stratified
        medium.

        Note that the function takes no argument in order to be used with
        dataset objects and ultimately with the fitter.

        :return: Q, scattered_intensity, (numpy arrays, abscissa and ordinate)
        """
        z, profile = self.create_electron_density_profile()
        N = int(2 ** np.ceil(np.log(1. / (self.dz * self.dQz))))

        freq = np.fft.fftfreq(N, self.dz)
        amp = np.fft.fft(profile, N)

        Q = np.arange(0, self.Qz_max, self.dQz)
        freq, amp = np.fft.fftshift(freq), np.fft.fftshift(amp)
        amp = np.interp(Q, freq, amp)
        background = 5
        I0 = self.parameter_controller.get_value('I0')
        intensity = np.array(np.abs(amp)**2, dtype=np.float) + background
        intensity = intensity / np.max(intensity) * I0

        return Q, intensity


def create_data(parameter_controller):
    """
    Create fake experimental data and the dataset, encapsulating these data and
    the functionality to simulate model data.

    :param parameter_controller: Instance of ParameterController
    :return: x_true, y_true, x_sampled, y_sampled, dataset
    """
    data = ReflectivityData(parameter_controller)
    x_true, y_true = data.fft_profile()
    y_true = y_true
    sampling_rate = 150
    x_sampled, y_sampled = x_true[::sampling_rate], y_true[::sampling_rate]
    y_sampled = poisson_error(y_sampled)
    y_sampled = np.maximum(y_sampled, 1)
    dataset = Dataset(x_sampled, y_sampled, sim_func=data.fft_profile)
    return x_true, y_true, x_sampled, y_sampled, dataset


def create_parameter_controller(
        num_sublayers=3, layer_repetitions=5, incident_intensity=5):
    """
    Create parameter controller and feed it a variable number of parameters,
    varying in whether they are supposed to be fitted or held constant, and
    sometimes being coupled to existing parameters.

    :param num_sublayers: Number of (randomised) sublayers forming a layer.
    :param layer_repetitions: Number of layers forming the multilayer.
    :param incident_intensity: Scaling factor for the incident intensity. The
        larger the incident intensity is, the smaller is the noise of the fake
        data.
    :return: parameter_controller
    """
    parameter_controller = ParameterController()

    # add some constant, which will be not be varied in the optimization process
    parameter_controller.add(
        parameter.Parameter('layer_reps', value=layer_repetitions, fit=False),
        parameter.Parameter('I0', value=10**incident_intensity, fit=False),
        )

    # initialise empty lists containing (randomised) parameters
    thickness_lst = []
    density_list = []

    # create an arbitrary number of sublayer thicknesses and densities
    for i in range(num_sublayers):
        bottom_thickness = np.random.uniform(3, 6)
        bottom_dens = np.random.uniform(5, 15)

        bottom_part = parameter.Parameter(
            f'd_{i}_bottom',
            value=bottom_thickness,
            bounds=(3, 6),
            fit=True,
        )

        bottom_dens = parameter.Parameter(
            f'e_dens_{i}_bottom', value=bottom_dens)

        # top part thickness is a constant fraction of the bottom  part thickness
        top_part = parameter.Parameter(
            f'd_{i}_top',
            value=0.66,
            fit=False,
            coupler=parameter.MultiplicativeCoupler(bottom_part),  # coupling
        )

        # top part density is equal to bottom part density plus a fixed value
        top_dens = parameter.Parameter(
            f'e_dens_{i}_top',
            value=3,
            fit=False,
            coupler=parameter.AdditiveCoupler(bottom_dens),  # coupling
        )

        # add parameters to controller to be accessed and fitted individually
        parameter_controller.add(bottom_part, top_part, bottom_dens, top_dens)

        # extend lists with parameters as group-constructor expects iterable
        thickness_lst.extend([bottom_part, top_part])
        density_list.extend([bottom_dens, top_dens])

    # create the Parameter Groups
    thickness_group = parameter.ParameterGroup('thicknesses', *thickness_lst)
    density_group = parameter.ParameterGroup('densities', *density_list)

    # Add the groups to the controller, to access list of values by group-name
    parameter_controller.add(thickness_group, density_group)

    return parameter_controller


# ==============================================================================
# ==============================================================================
# ==============================================================================


# (random) sub-layers per layer. Higher values mean more complex structure
num_sublayers = 3

# Number of multilayer repetitions. Higher numbers means stronger defined signal
layer_repetitions = 8

# Scaling factor determining the intensity of incident radiation.
# Higher numbers means less noise, change in integer steps
incident_intensity = 4

# Scattering data are often better represented in logarithmic fashion
# Logarithmic figure of merit is more sensitive to low intensity data
use_logarithmic_fom = True


# ==============================================================================
# ==============================================================================
# ==============================================================================


parameter_controller = create_parameter_controller(
    num_sublayers=num_sublayers,
    layer_repetitions=layer_repetitions,
    incident_intensity=incident_intensity,
    )


x_true, y_true, x_sampled, y_sampled, dataset = create_data(parameter_controller)

fitter = Fitter(parameter_controller)
fom_type = 'log' if use_logarithmic_fom else 'diff'
fitter.add_dataset(dataset,  fom_type=fom_type)
result = fitter.optimize()

print(result)
x_fit, y_fit = dataset.simulate()
fom_array = fitter.fom_handler.fom_arrays[0]

z, profile = ReflectivityData(parameter_controller).create_electron_density_profile()


# ==============================================================================
# ==============================================================================
# ==============================================================================


fig1, ax = plt.subplots()
ax.plot(z, profile, color='k')
ax.set_ylabel('Density profile of multilayer [arb units]')
ax.set_xlabel('Sample height z [nm]')

fig2, [ax1, ax2] = plt.subplots(
    nrows=2, sharex=True, gridspec_kw={'hspace': 0})
ax1.plot(x_true, y_true, color='k', linestyle='--')
ax1.scatter(x_sampled, y_sampled, marker='x', c='k')
ax1.plot(x_fit, y_fit, color='g')
if  use_logarithmic_fom:
    ax1.set_yscale('log')
ax1.set_ylabel('Intensity [detector counts]')

ax2.plot(x_fit, fom_array, color='r')
ax2.set_ylabel(f'"{fom_type}" figure of merit')
ax2.set_xlabel(r'Wavevector transfer Q [nm$^{-1}$]')

plt.show()

import numpy as np
import matplotlib.pyplot as plt

from xray_diffraction import ParameterController, Fitter, Dataset
from xray_diffraction import parameter


def poisson_error(y):
    return y + np.random.normal(scale=np.sqrt(y))


def make_bilayer(z, z0, d1, d2):
    e_dens_1, e_dens_2 = 15, 5
    layer1 = np.where(np.logical_and(z>z0, z<=z0+d1), e_dens_1, 0)
    layer2 = np.where(np.logical_and(z>z0+d1, z<=z0+d1+d2), e_dens_2, 0)
    return layer1 + layer2


def make_stack(z, d1, d2, num_rep):
    d_bilayer = d1+d2
    profile = np.zeros_like(z)
    for i in range(num_rep):
        bilayer = make_bilayer(z=z, z0=i*d_bilayer, d1=d1, d2=d2)
        profile = profile + bilayer
    return profile


class ReflectivityData:
    def __init__(self, parameter_controller):
        self.dz = 0.01
        self.dQz = 0.00001
        self.Qz_max = 0.25
        self.z = np.arange(0, 100, self.dz)
        self.parameter_controller = parameter_controller

    def create_electron_density_profile(self):
        d1 = self.parameter_controller.get_value('d1')
        d2 = self.parameter_controller.get_value('d2')
        profile = make_stack(self.z, d1, d2, num_rep=10)
        return self.z, profile

    def fft_profile(self):
        z, profile = self.create_electron_density_profile()
        N = int(2 ** np.ceil(np.log(1. / (self.dz * self.dQz))))

        freq = np.fft.fftfreq(N, self.dz)
        amp = np.fft.fft(profile, N)

        Q = np.arange(0, self.Qz_max, self.dQz)
        freq, amp = np.fft.fftshift(freq), np.fft.fftshift(amp)
        amp = np.interp(Q, freq, amp)
        background = 5
        intensity = 1e-7 * np.array(np.abs(amp)**2, dtype=np.float) + background

        return Q, intensity


def create_data():
    data = ReflectivityData(parameter_controller)
    x_true, y_true = data.fft_profile()
    y_true = y_true
    sampling_rate = 150
    x_sampled, y_sampled = x_true[::sampling_rate], y_true[::sampling_rate]
    y_sampled = poisson_error(y_sampled)
    y_sampled = np.maximum(y_sampled, 1)
    dataset = Dataset(x_sampled, y_sampled, sim_func=data.fft_profile)
    return x_true, y_true, x_sampled, y_sampled, dataset


parameter_controller = ParameterController()

parameter_controller.add(
    parameter.Parameter(
        'd1',
        value=3.5,
        bounds=(1, 4),
        fit=True,
        ),
)
if True:
    parameter_controller.add(
        parameter.Parameter(
            'd2',
            value=6,
            bounds=(4, 8),
            fit=True,
        )
    )
else:
    parameter_controller.add(
        parameter.Parameter(
            'bilayer_thickness', value=9.5, bounds=(7., 11.), fit=True)
    )


data = ReflectivityData(parameter_controller)
z, profile = data.create_electron_density_profile()
qz, amp = data.fft_profile()

x_true, y_true, x_sampled, y_sampled, dataset = create_data()

fitter = Fitter(parameter_controller)
fitter.add_dataset(dataset,  fom_type='diff')
result = fitter.optimize()
print(result)
x_fit, y_fit = dataset.simulate()
fom_array = fitter.fom_handler.fom_arrays[0]

fig, [ax1, ax2] = plt.subplots(
    nrows=2, sharex=True, gridspec_kw={'wspace': 0})
ax1.plot(x_true, y_true)
ax1.scatter(x_sampled, y_sampled, marker='x')
ax1.plot(x_fit, y_fit)
ax1.set_yscale('log')

ax2.plot(x_fit, fom_array)

plt.show()

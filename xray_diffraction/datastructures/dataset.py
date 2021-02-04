import numpy as np


class Dataset(object):
    """
    Container class for experimental data and functions for simulating data.
    """
    def __init__(self, x=None, y=None, sim_func=None,
                 bkg=None, x_label='x', y_label='y', info=None, error=None,
                 ):
        """
        :param x: array-like abscissa values of experimental data
        :param y: array-like ordinate values of experimental data
        :param sim_func: argument-less function, returning simulated x- and
            y-data upon being called. If simulated x-values are not identical to
            experimental x-values, simulated data will be interpolated in order
            to allow point-by-point comparison of experimental and simulated
            data
        :param bkg: array-like background signal not captured by the sim_func
            added to the simulated y-data
        :param x_label: string, label of the abscissa axis
        :param y_label: string, label of the ordinate axis
        :param info: string, additional info
        :param error: array-like same size as experimental x and y. Associated
            errors of the experimental data-points
        """
        self.bkg = bkg
        self.x, self.y = np.array(x), np.array(y)
        self.x_sim, self.y_sim = None, None
        self.sim_func = sim_func
        self.x_label, self.y_label = x_label, y_label
        self.info = info if info else {}
        self.mask = None
        self._init_mask()
        self.error = error

    @property
    def num_masked(self):
        return np.sum(self.mask)

    def _interpolation_necessary(self):
        return len(self.y_sim) != len(self.y)

    def _interpolate_data(self):
        if len(self.y_sim) >= len(self.y):
            self.y_sim = np.interp(self.x, self.x_sim, self.y_sim)
            self.x_sim = self.x
        else:
            self.y = np.interp(self.x_sim, self.x, self.y)
            self.x = self.x_sim

    def simulate(self):
        """
        Simulates data usually trying to model the experimental data
        :return: x_sim-array, y_sim-array
        """
        x_sim, y_sim = self.sim_func()
        self.x_sim, self.y_sim = np.array(x_sim), np.array(y_sim)
        if self._interpolation_necessary():
            self._interpolate_data()
        if self.bkg is not None:
            self.y_sim += self.bkg
        return self.x_sim, self.y_sim

    def _init_mask(self):
        self.mask = np.full_like(self.x, False, dtype=bool)
        return self

    def mask_above(self, limit):
        """
        Create a mask, usually indicating data points above a certain x-value to
        be ignored in subsequent analysis

        :param limit: float, value above which data will be masked
        :return: None
        """
        self.mask = np.where(self.x > limit, True, self.mask)
        return self

    def mask_below(self, limit):
        """
        Create a mask, usually indicating data points below a certain x-value to
        be ignored in subsequent analysis

        :param limit: float, value below which data will be masked
        :return: None
        """
        self.mask = np.where(self.x < limit, True, self.mask)
        return self

    def clear_mask(self):
        """
        Remove any previous masking of the data.

        :return: None
        """
        self._init_mask()
        return self

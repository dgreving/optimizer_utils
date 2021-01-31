import numpy as np


class Dataset(object):
    def __init__(self, x=None, y=None, sim_func=None,
                 bkg=None, x_label='x', y_label='y', info=None, error=None,
                 ):
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
        self.mask = np.where(self.x > limit, True, self.mask)
        return self

    def mask_below(self, limit):
        self.mask = np.where(self.x < limit, True, self.mask)
        return self

    def clear_mask(self):
        self._init_mask()
        return self

from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
from xray_diffraction.datastructures import fom_funcs
from xray_diffraction.datastructures.fom_funcs import handle_masked_FOM


class FOMCalculator:

    _map = {
        'diff': {'FOM': fom_funcs.FOM_diff, 'array_func': fom_funcs.diff_array},
        'diff_norm': {'FOM': fom_funcs.FOM_diff, 'array_func': fom_funcs.diff_norm_array},
        'log': {'FOM': fom_funcs.FOM_log, 'array_func': fom_funcs.log_array},
        'R1': {'FOM': fom_funcs.FOM_R1, 'array_func': fom_funcs.R1_array},
        'R1_log': {'FOM': fom_funcs.FOM_R1_log, 'array_func': fom_funcs.R1_log_array},
        'R2': {'FOM': fom_funcs.FOM_R2, 'array_func': fom_funcs.R2_array},
        'R2_log': {'FOM': fom_funcs.FOM_R2_log, 'array_func': fom_funcs.R2_log_array},
        'log_rangeNorm': {'FOM': fom_funcs.FOM_diff, 'array_func': fom_funcs.log_rangeNorm_array},
        'diff_rangeNorm': {'FOM': fom_funcs.FOM_diff, 'array_func': fom_funcs.diff_rangeNorm_array},
        'chi2': {'FOM': fom_funcs.FOM_diff, 'array_func': fom_funcs.chi2_array},
        'chi': {'FOM': fom_funcs.FOM_diff, 'array_func': fom_funcs.chi_array},
        }

    def __init__(self, dataset, fom_type):
        self.dataset = dataset
        self.fom_type = fom_type
        self._array_func = self._map[fom_type]['array_func']
        self._fom_func = self._map[fom_type]['FOM']
        self.x_sim = None
        self.y_sim = None
        self.fom_array = None
        self.fom = None
        self.calc()

    def calc(self):
        self.x_sim, self.y_sim = self.dataset.simulate()
        self.fom_array = self._create_fom_array()
        self.fom = self._calc_fom_from_array(self.fom_array)

    def _create_fom_array(self):
        arr = self._array_func(self.y_sim, self.dataset)
        return handle_masked_FOM(arr, self.dataset)

    def _calc_fom_from_array(self, array):
        return self._fom_func(array, self.dataset)


class FOMHandler:
    def __init__(self):
        self.preprocessor_funcs = []
        self.datasets = []
        self.x_sims = []
        self.y_sims = []
        self.fom_arrays = []
        self.foms = []
        self.composite_fom = None

    @property
    def num_active_fits(self):
        return sum([ds['fit'] for ds in self.datasets])

    def add_preprocessor(self, preprocessor_func):
        self.preprocessor_funcs.append(preprocessor_func)

    def add_dataset(self, dataset, fom_type='diff', fit=True):
        self.datasets.append(dict(dataset=dataset, fom_type=fom_type, fit=fit))
        return self

    def calc(self):
        for preprocess in self.preprocessor_funcs:
            preprocess()
        fit_flags = [ds['fit'] for ds in self.datasets]
        calcs = [
            FOMCalculator(dataset=ds['dataset'], fom_type=ds['fom_type'])
            for ds in self.datasets
        ]
        self.x_sims = [c.x_sim for c in calcs]
        self.y_sims = [c.y_sim for c in calcs]
        self.fom_arrays = [
            c.fom_array if fit else None for c, fit in zip(calcs,  fit_flags)
            ]
        self.foms = [c.fom if fit else 0 for c, fit in zip(calcs, fit_flags)]
        self.composite_fom = sum(self.foms) / sum(fit_flags)


class Fitter(object):
    def __init__(self, master_controller, algorithm='DE', **algo_kwargs):
        self.master_controller = master_controller
        self.algorithm_type = algorithm
        self.algo_kwargs = algo_kwargs
        self.solver = None
        self.fit_callback = None

        self.fom_handler = FOMHandler()

        self._solvers = dict(
            DE=self.create_DE_solver,
            )

    @property
    def fit_keys(self):
        return list(self.master_controller.keys(only_fitted=True))

    def update_master_controller(self, keys, vals):
        self.master_controller.update(
            *[(key, val) for key, val in zip(keys, vals)]
        )

    def add_preprocessor(self, preprocess_func):
        # print("adding preprocess: ", preprocess_func.__name__)
        self.fom_handler.add_preprocessor(preprocess_func)

    def add_dataset(self, dataset, fit=True, fom_type='diff'):
        self.fom_handler.add_dataset(dataset, fom_type, fit)
        return self

    def set_fit_callback(self, fit_callback):
        self.fit_callback = fit_callback

    def fom_func(self, fit_vals):
        self.update_master_controller(self.fit_keys, fit_vals)
        self.fom_handler.calc()
        return self.fom_handler.composite_fom

    def optimize(self):
        solver = self.create_solver()
        result = solver.solve()
        optimal_vals = result.x
        self.fom_func(optimal_vals)
        return result

    def simulate(self, controller=None):
        vals = self.master_controller.extract_vals()
        self.fit_keys, self.fit_bounds = controller.extract_keys_and_bounds()
        self.perform_preprocessing()
        self.fit_callback(xk=vals, convergence=None, sim_only=True)

    def create_solver(self):
        return self._solvers[self.algorithm_type]()

    def create_DE_solver(self):
        return DifferentialEvolutionSolver(
            self.fom_func,
            list(self.master_controller.bounds(only_fitted=True)),
            callback=self.fit_callback,
            **self.algo_kwargs)

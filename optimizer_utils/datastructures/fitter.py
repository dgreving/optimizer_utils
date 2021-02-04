from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
from optimizer_utils.datastructures import fom_funcs
from optimizer_utils.datastructures.fom_funcs import handle_masked_FOM


class FOMCalculator:
    """
    Provide functionality for calculating a figure of merit, measuring the
    similarity of arrays of data points. Data is generally provided by an
    instance of a "Dataset" class, usually containing experimental data and a
    way of generating arrays of simulated data.
    """

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

    def __init__(self, dataset, fom_type='diff'):
        """
        :param dataset: Instance of Dataset class, providing experimental and
        simulated data.
        :param fom_type: String identifier, determining which function is being
        used to generate the figure of merit.
        """
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
        """
        Calculate simulated data, a FOM-array and the FOM, which is a collapsed
        version of the FOM-array, usually obtained by some form of summation
        over the FOM-array.
        :return: None
        """
        self.x_sim, self.y_sim = self.dataset.simulate()
        self.fom_array = self._create_fom_array()
        self.fom = self._calc_fom_from_array(self.fom_array)

    def _create_fom_array(self):
        arr = self._array_func(self.y_sim, self.dataset)
        return handle_masked_FOM(arr, self.dataset)

    def _calc_fom_from_array(self, array):
        return self._fom_func(array, self.dataset)


class FOMHandler:
    """
    Provides functionality to handle a number of Dataset instances and to
    calculate their composite figure of merit, i.e. the average FOM of all
    datasets that have their "fit" flat set to True.
    """
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
        """
        Adds a preprocessing function that is invoked every time before the
        FOM is calculated by simulating data from within the dataset objects.

        :param preprocessor_func: Any callable taken no arguments. May often
        be used to perform some updating of data before simulations are
        performed.
        :return: None
        """
        self.preprocessor_funcs.append(preprocessor_func)

    def add_dataset(self, dataset, fom_type='diff', fit=True):
        """
        Add single dataset to the handler.

        :param dataset: instance of dataset object
        :param fom_type: string, identifying the type of function used to
            calculate the figure of merit
        :param fit: Boolean flag, indicating whether the dataset is considered
            in calculating the composite FOM
        :return: None
        """
        self.datasets.append(dict(dataset=dataset, fom_type=fom_type, fit=fit))
        return self

    def calc(self):
        """
        Calculates x_sims, y_sims, fom_arrays, individual FOMs and composite FOM
        of _all_ datasets, but uses only fitted datasets to calculate the
        composite FOM.

        :return: None
        """
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


class Fitter:
    """
    Providing functionality to simultaneously fit multiple sets of data by
    variation of a set of parameters contained in a parameter_controller.
    """
    def __init__(self, master_controller, algorithm='DE', **algo_kwargs):
        self.master_controller = master_controller
        self.algorithm_type = algorithm
        self.algo_kwargs = algo_kwargs
        self.fit_callback = None

        self.fom_handler = FOMHandler()

        self._solvers = dict(
            DE=self._create_DE_solver,
            )

    @property
    def fit_keys(self):
        """Return a list of the names of all fitted parameters"""
        return list(self.master_controller.keys(only_fitted=True))

    def _update_master_controller(self, keys, vals):
        """
        :param keys: iterable containing all keys of parameters to update values
        :param vals: iterable containing the new values of the updated paraeters
        :return: None
        """
        self.master_controller.update(
            *[(key, val) for key, val in zip(keys, vals)]
        )

    def add_preprocessor(self, preprocess_func):
        """
        Adds a preprocessing function that is invoked every time before the
        FOM is calculated by simulating data from within the dataset objects.

        :param preprocessor_func: Any callable taken no arguments. May often
        be used to perform some updating of data before simulations are
        performed.
        :return: None
        """
        self.fom_handler.add_preprocessor(preprocess_func)

    def add_dataset(self, dataset, fit=True, fom_type='diff'):
        """
        Add single dataset to the internal dataset handler of the fitter.

        :param dataset: instance of dataset object
        :param fom_type: string, identifying the type of function used to
            calculate the figure of merit
        :param fit: Boolean flag, indicating whether the dataset is considered
            in calculating the composite FOM
        :return: None
        """
        self.fom_handler.add_dataset(dataset, fom_type, fit)
        return self

    def set_fit_callback(self, fit_callback):
        """
        Add a function that is invoked at each optimization step

        :param fit_callback: callable
        :return: None
        """
        self.fit_callback = fit_callback

    def _fom_func(self, fit_vals):
        """
        Calculate the composite figure of merit , as a function of an iterable
        of new values of the fitted parameters.

        :param fit_vals: Iterable, new value of each fitted parameter
        :return: Composite figure of merit of all fitted datasets
        """
        self._update_master_controller(self.fit_keys, fit_vals)
        self.fom_handler.calc()
        return self.fom_handler.composite_fom

    def optimize(self):
        """
        Optimize all fitted datasets under the current state of the fitter.

        :return: return a result object as obtained by the used solver of the
            optimization process
        """
        solver = self._create_solver()
        result = solver.solve()
        optimal_vals = result.x
        self._fom_func(optimal_vals)
        return result

    def simulate(self, controller=None):
        raise NotImplementedError()
        self.perform_preprocessing()
        self.fit_callback(xk=vals, convergence=None, sim_only=True)

    def _create_solver(self):
        return self._solvers[self.algorithm_type]()

    def _create_DE_solver(self):
        return DifferentialEvolutionSolver(
            self._fom_func,
            list(self.master_controller.bounds(only_fitted=True)),
            callback=self.fit_callback,
            **self.algo_kwargs)

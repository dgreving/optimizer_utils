import pytest
import numpy as np

from xray_diffraction.datastructures.dataset import Dataset
from xray_diffraction.datastructures.parameter import Parameter
from xray_diffraction.datastructures.parameter_controller import (
    ParameterController)
from xray_diffraction.datastructures.fitter import (
    Fitter, FOMHandler, FOMCalculator)


@pytest.fixture
def datasets():
    x = [1, 2, 3]
    sim_func = lambda: (x, [_**2 for _ in x])
    dataset1 = Dataset(x=x, y=[1, 4, 9], sim_func=sim_func)
    dataset2 = Dataset(x=[1, 2, 3], y=[1, 8, 27], sim_func=sim_func)
    dataset3 = Dataset(x=[1, 2, 3], y=[1, 8, 27], sim_func=sim_func)
    dataset3.mask_above(limit=2)
    return dataset1, dataset2, dataset3


@pytest.fixture
def preprocessor():
    class TestPreprocessor:
        def __init__(self):
            self.running_index = 0

        def increment(self):
            self.running_index += 1

    return TestPreprocessor()


class TestFOMCalculator:

    def test_calc(self, datasets):
        expected_foms = [0./3, 22./3, 4./2]
        for ds, expected_fom in zip(datasets, expected_foms):
            mask = ds.mask
            calculator = FOMCalculator(ds, 'diff')
            y_exp, y_sim = ds.y, calculator.y_sim
            x_sim = calculator.x_sim
            fom_arr = calculator.fom_array
            fom = calculator.fom
            for this_fom, this_mask, yX, yS in zip(fom_arr, mask, y_exp, y_sim):
                if this_mask is True:
                    assert this_fom == 0
                else:
                    this_fom == np.abs(yX - yS)
            assert fom == expected_fom
            assert all(x_orig == x_sim for x_orig, x_sim in zip(ds.x, x_sim))
            assert len(y_sim) == 3
            assert len(fom_arr) == 3
            assert type(fom) != 0


class TestFOMHandler:

    def test_add_datasets(self, datasets):
        fh = FOMHandler()
        ds1, ds2, ds3 = datasets[:3]
        fh.add_dataset(ds1).add_dataset(ds2).add_dataset(ds3, fit=False)
        fh.calc()
        assert len(fh.datasets) == 3
        assert fh.datasets[0]['dataset'] == ds1
        assert fh.datasets[0]['fom_type'] == 'diff'

    def test_composite_FOM(self, datasets):
        fh = FOMHandler()
        ds1, ds2, ds3 = datasets[:3]
        fh.add_dataset(ds1).add_dataset(ds2).add_dataset(ds3, fit=False)
        fh.calc()
        expected_composite_fom = (0 + 22./3) / fh.num_active_fits
        assert fh.num_active_fits == 2
        assert fh.composite_fom == expected_composite_fom

    def test_preprocessing(self, datasets, preprocessor):
        fh = FOMHandler()
        fh.add_preprocessor(preprocessor.increment)
        fh.add_dataset(datasets[0]).add_dataset(datasets[1])
        assert preprocessor.running_index == 0
        fh.calc()
        assert preprocessor.running_index == 1


class TestFitter:

    @pytest.fixture
    def master_controller(self):
        p1 = Parameter(name='p1', value=4, bounds=(0, 5), fit=True)
        p2 = Parameter(name='p2', value=2, bounds=(0, 10), fit=True)
        p3 = Parameter(name='p3', value=3, fit=False)
        controller = ParameterController()
        controller.add(p1, p2, p3)
        return controller

    @pytest.fixture
    def fit_datasets(self, master_controller):
        mc = master_controller

        def f1():
            x = [0]
            y = [mc.get_value('p1') + mc.get_value('p2') + mc.get_value('p3')]
            return x, y

        def f2():
            x = [0]
            y = [mc.get_value('p1') - mc.get_value('p2') + mc.get_value('p3')]
            return x, y

        d1 = Dataset(x=[0], y=[8], sim_func=f1)
        d2 = Dataset(x=[0], y=[2], sim_func=f2)
        return d1, d2

    def test_optimize(self, master_controller, fit_datasets, preprocessor):
        d1, d2 = fit_datasets
        fitter = Fitter(master_controller, algorithm='DE')
        fitter.add_preprocessor(preprocessor.increment)
        fitter.add_dataset(d1, fit=True, fom_type='diff')
        fitter.add_dataset(d2, fit=True, fom_type='diff')
        assert preprocessor.running_index == 0  # test preprocessing not called
        result = fitter.optimize()
        expected = [
            master_controller.get_value('p1'),
            master_controller.get_value('p2'),
        ]
        assert all(np.isclose(val, exp) for val, exp in zip(result.x, expected))
        # test preprocessing was called for each function evaluation
        assert preprocessor.running_index == result.nfev


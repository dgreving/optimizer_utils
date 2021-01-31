import numpy as np
import pytest
from xray_diffraction.datastructures.dataset import Dataset

x = np.linspace(0, 3, 4)
y = x ** 2.0
x_sim = np.linspace(0, 3, 8)
same_len_func = lambda: (x, x**2)
unequal_len_func = lambda: (x_sim, x_sim**2)

@pytest.fixture
def dataset():
    dataset = Dataset(
        x=x,
        y=y,
        sim_func=same_len_func,
        x_label='time',
        y_label='amplitude',
        info='rnd_info',
        error=np.sqrt(y),
    )
    return dataset


@pytest.mark.parametrize(
    'test_func,bkg',
    [
        (same_len_func, 0),
        (unequal_len_func, 0),
        (same_len_func, 0.3),
        (unequal_len_func, 0.3),
    ])
def test_simulate(test_func, bkg, dataset):
    dataset.sim_func = test_func
    if bkg:
        dataset.bkg = np.full_like(dataset.x, bkg)
    x, y = dataset.x, dataset.y
    x_sim, y_sim = dataset.simulate()
    assert all(x1 == x2 for x1, x2 in zip(x, x_sim))
    assert len(y) == len(y_sim)
    assert all(np.isclose(y + bkg, y_sim, rtol=0.05))  # interpol. errors


def test_mask_above(dataset):
    limit = 1
    assert dataset.mask_above(limit=limit) is dataset  # test chains
    for xs, masked in zip(x, dataset.mask):
        if xs > limit:
            assert (masked == True)
        elif xs <= limit:
            assert (masked == False)  # 'masked is False' fails?!


def test_mask_below(dataset):
    limit = 1
    assert dataset.mask_below(limit=limit) is dataset  # test chains
    for xs, masked in zip(x, dataset.mask):
        if xs >= limit:
            assert (masked == False)
        elif xs < limit:
            assert (masked == True)


def test_clear_mask(dataset):
    limit = 1
    assert dataset.mask_below(limit=limit) is dataset  # test chains
    dataset.clear_mask()
    assert all(masked == False for masked in dataset.mask)
    # don't know why masked is False does not work...


def test_property_num_masked(dataset):
    limit = 1
    dataset.mask_above(limit=limit)
    assert dataset.num_masked == 2  # x == [0, 1, 2, 3]



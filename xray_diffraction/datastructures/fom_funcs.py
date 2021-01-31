import numpy as np


def handle_masked_FOM(FOM_array, dataset):
    if dataset.mask is None:
        return FOM_array
    else:
        return np.where(dataset.mask, 0., FOM_array)


def diff_array(y_s, dataset):
    array = dataset.y - y_s
    return handle_masked_FOM(array, dataset)


def diff_norm_array(y_s, dataset):
    y = dataset.y
    array = np.minimum(1.5, np.abs((y - y_s) / np.maximum(1., np.abs(y))))
    return handle_masked_FOM(array, dataset)


def diff_rangeNorm_array(y_s, dataset):
    range_min = np.min(np.where(dataset.mask, np.inf, dataset.y))
    range_max = np.max(np.where(dataset.mask, -np.inf, dataset.y))
    # range_min, range_max = np.min(dataset.y), np.max(dataset.y)
    array = (dataset.y - y_s) / (range_max - range_min)
    return handle_masked_FOM(array, dataset)


def log_array(y_s, dataset):
    array = np.log10(dataset.y) - np.log10(y_s)
    return handle_masked_FOM(array, dataset)


def R1_array(y_s, dataset):
    s = np.sign
    y = dataset.y
    array = np.abs(s(y) * np.sqrt(np.abs(y)) - s(y_s) * np.sqrt(np.abs(y_s)))
    return handle_masked_FOM(array, dataset)


def log_rangeNorm_array(y_s, dataset):
    y_s, y = np.abs(y_s) + 10, np.abs(dataset.y) + 10
    array = (np.log10(y_s) - np.log10(y)) / (np.log10(np.max(y)) - np.log10(np.min(y)))
    return handle_masked_FOM(array, dataset)


def FOM_R1(array, dataset):
    return np.sum(array) / np.sum(np.sqrt(handle_masked_FOM(dataset.y, dataset)))  # nopep8


def R1_log_array(y_s, dataset):
    array = np.abs(np.log10(np.sqrt(dataset.y)) - np.log10(np.sqrt(y_s)))
    return handle_masked_FOM(array, dataset)


def FOM_R1_log(array, dataset):
    total_sum = np.sum(np.log10(np.sqrt(handle_masked_FOM(dataset.y, dataset))))  # nopep8
    return np.sum(array) / total_sum


def R2_array(y_s, dataset):
    array = (dataset.y - y_s)**2
    return handle_masked_FOM(array, dataset)


def FOM_R2(array, dataset):
    return np.sum(array) / np.sum(handle_masked_FOM(dataset.y, dataset)**2)


def R2_log_array(y_s, dataset):
    y = dataset.y
    array = (np.log10(y) - np.log10(y_s))**2
    return handle_masked_FOM(array, dataset)


def FOM_R2_log(array, dataset):
    total_sum = np.sum(handle_masked_FOM(np.log10(dataset.y)**2, dataset))
    return np.sum(array) / total_sum


def chi2_array(y_s, dataset):
    array = ((dataset.y - y_s) / dataset.error)**2
    return handle_masked_FOM(array, dataset)


def chi_array(y_s, dataset):
    array = ((dataset.y - y_s) / dataset.error)
    return handle_masked_FOM(array, dataset)


def FOM_diff(array, dataset):
    numMasked = dataset.num_masked
    return np.sum(np.abs(array)) / (len(array) - numMasked)


def FOM_log(array, dataset):
    if dataset.parameters.__class__.__name__ == 'ParameterController':
        numFreeParameters = dataset.parameters.num_fitted_parameters
    else:
        numFreeParameters = len(dataset.parameters)
        print("""Warning, all parameters provided to dataset are considered
            free. Consider providing parameters in ParameterController object
            to ensure only parameters with attribute 'fit'==True are
            contributing to free parameters.""")
    numMasked = dataset.num_maskedPoints
    return np.sum(np.abs(array)) / (len(array) - numMasked - numFreeParameters)


def FOM_sum(array, dataset):
    return np.sum(array)


def diff_rangeNorm_special(y_s, y):
    array = (y - y_s)**2 / y**2
    return array

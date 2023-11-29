import logging
import traceback

import numpy as np
from scipy.interpolate import interp1d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_func_length(combined_train, func, second_max=False):
    try:
        if func == min:
            func_length = np.inf
            second = np.inf
        else:
            func_length = 0
            second = 0

        n = combined_train.shape[0]
        for i in range(n):
            temp_max = func(func_length, combined_train[i].shape[0])
            if temp_max > func_length:
                second = func_length
                func_length = temp_max
            elif second_max and temp_max > second and temp_max != func_length:
                second = temp_max
        if second_max:
            logger.info("Second max is: {}".format(second))
            return second
        return func_length
    except Exception as e:
        logger.info("Error getting the function length: {}".format(str(e)))


def pad_to_same_length(x, n_var, max_length):
    try:
        n = x.shape[0]
        interpolated_data = np.zeros((n, max_length, n_var), dtype=np.float64)
        for i in range(n):
            mts = x[i]
            # TODO count for the pids
            # pid = mts[0][-1]
            # mts = mts[:, :-1]
            shape = np.shape(mts)
            interpolated_data[i, :shape[0], :shape[1]] = mts
            # interpolated_data[i, :, -1] = pid
        return interpolated_data
    except Exception as e:
        logger.error("Error in transforming the data to the same length: {}".format(str(e)))


def transform_to_same_length(x, n_var, max_length, padding=False):
    if padding:
        return pad_to_same_length(x, n_var, max_length)
    try:
        n = x.shape[0]
        interpolated_data = np.zeros((n, max_length, n_var), dtype=np.float64)

        for i in range(n):
            mts = x[i]
            curr_length = mts.shape[0]
            idx = np.array(range(curr_length))
            idx_new = np.linspace(0, idx.max(), max_length)
            # TODO count for the pids
            # pid = mts[0][-1]
            for j in range(n_var):
                ts = mts[:, j]
                # linear interpolation
                f = interp1d(idx, ts, kind='cubic')
                new_ts = f(idx_new)
                interpolated_data[i, :, j] = new_ts
            # interpolated_data[i, :, -1] = pid
        return interpolated_data
    except Exception as e:
        logger.error("Error in transforming the data to the same length: {}".format(str(e)))
        logger.info(traceback.format_exc())


def interpolate_coordinates(combined_data, max_length, padding):
    interpolated_data = np.array([])
    try:
        # TODO count for the pids
        n_var = combined_data[0].shape[1]
        logger.info("Maximum length: {}, number of dimensions: {}".format(str(max_length), str(n_var)))
        if len(combined_data):
            interpolated_data = transform_to_same_length(combined_data, n_var, max_length, padding)
    except Exception as e:
        logger.error("Error performing the interpolation: {}".format(str(e)))
    return interpolated_data


def create_mr_seql_args(x_array):
    maximum_length = get_func_length(x_array, func=max)
    minimum_length = get_func_length(x_array, func=min)

    mr_seql_format_args = {
        "number_of_dimensions": x_array[0].shape[1],
        "minimum_length": minimum_length,
        "maximum_length": maximum_length,
        "number_of_time_series": x_array.shape[0]
    }

    return mr_seql_format_args


def create_sktime_args(x_array):
    maximum_length = get_func_length(x_array, func=max)
    minimum_length = get_func_length(x_array, func=min)

    sktime_format_args = {
        "number_of_dimensions": x_array[0].shape[1],
        "minimum_length": minimum_length,
        "maximum_length": maximum_length,
        "number_of_time_series": x_array.shape[0]
    }

    return sktime_format_args

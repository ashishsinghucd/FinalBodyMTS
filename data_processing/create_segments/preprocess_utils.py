import pandas as pd
import numpy as np
import peakutils
import logging
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelextrema, find_peaks, peak_prominences

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def smooth_coordinates_sf(raw_coordinates):
    """
    Smooth the coordinates using the savgol filter
    """
    try:
        smooth_coordinates = savgol_filter(raw_coordinates, 31, 3)
        return smooth_coordinates
    except Exception as e:
        logger.error("Error in smoothening the signal: {}".format(str(e)))
        return None


def remove_baseline(coordinates):
    """
    Function to remove the baseline
    """
    try:
        baseline = peakutils.baseline(coordinates, 5)
        scaled = coordinates - baseline
        return scaled
    except Exception as e:
        logger.error("Error removing the baseline: {}".format(str(e)))
        return None


def identify_best_body_part(df, test_columns):
    """
    Function to try all combinations of body parts to get the minimum number of peaks
    """
    from itertools import combinations
    combination_list = []
    for i in range(len(test_columns)):
        oc = combinations(test_columns, i + 1)
        for column_combination in oc:
            combination_list.append(list(column_combination))
    # print("Column combinations: ", combination_list)
    best_body_part_list = []
    peak_length_list = []
    for i, column_combination in enumerate(combination_list):
        smooth_coordinates = df.loc[:, column_combination].sum(axis=1)
        smooth_coordinates = smooth_coordinates.to_numpy()
        peaks_max_y, _ = find_peaks(smooth_coordinates)
        peak_length_list.append(len(peaks_max_y))
    minimum_peaks = min(peak_length_list)

    # print("Minimum peaks: ", minimum_peaks)

    minimum_peaks_list = [index for index, element in enumerate(peak_length_list) if minimum_peaks == element]

    for i in minimum_peaks_list:
        best_body_part_list.append(combination_list[i])

    logger.info("Best body parts: {}".format(best_body_part_list[0]))

    return best_body_part_list[0]


def get_peaks(df, combined_columns, pid, exercise_type, threshold=10, plot_peaks=False):
    # combined_columns = ["RElbow"]
    best_columns = identify_best_body_part(df, combined_columns)
    # best_columns = ["RElbow"]
    best_columns_str = ",".join(best_columns)
    try:
        smooth_coordinates = df.loc[:, best_columns].sum(axis=1)
        smooth_coordinates = smooth_coordinates.to_numpy()
        peaks_max_y, _ = find_peaks(smooth_coordinates)

        prominences = peak_prominences(smooth_coordinates, peaks_max_y)[0]
        contour_heights = smooth_coordinates[peaks_max_y] - prominences
        heights = smooth_coordinates[peaks_max_y] - contour_heights
        filtered_peaks_max = []
        for i in np.arange(len(peaks_max_y)):
            if heights[i] < threshold:
                continue
            filtered_peaks_max.append(peaks_max_y[i])

        logger.info("Intial length of peaks list: {}".format(len(filtered_peaks_max)))
        # if len(filtered_peaks_max) == 10:
        #     filtered_peaks_max = [0] + filtered_peaks_max
        # elif len(filtered_peaks_max) == 9:
        #     filtered_peaks_max = [0] + filtered_peaks_max + [len(smooth_coordinates) - 1]
        filtered_peaks_max = np.array(filtered_peaks_max)
        if plot_peaks:
            index = np.arange(df.shape[0])
            _ = plt.plot(index, smooth_coordinates, 'b-', linewidth=2)
            _ = plt.plot(index[filtered_peaks_max], smooth_coordinates[filtered_peaks_max], 'ro')
            _ = plt.title(
                pid + ":" + best_columns_str + "{}".format(exercise_type) + "peaks: " + str(len(filtered_peaks_max)))
            plt.tight_layout()
            plt.savefig("/tmp/{}_{}_peaks.jpg".format(pid, exercise_type))
            plt.close()
        return filtered_peaks_max
    except Exception as e:
        logger.error("Error in getting the peaks: {}".format(str(e)))


# def plot_body_parts(df, pid, exercise_type, body_parts):
#     # body_parts = ["RElbow", "LElbow", "RWrist", "LWrist"]
#     fig = plt.figure(figsize=(15, 5))
#     count = 0
#     for body_part in body_parts:
#         ax = fig.add_subplot(2, 2, count + 1)
#         smooth_coordinates = smooth_coordinates_sf(df[body_part])
#         smooth_coordinates_std = (smooth_coordinates - smooth_coordinates.mean()) / np.std(smooth_coordinates)
#         _ = ax.plot(smooth_coordinates_std)
#         _ = ax.set_title("{}_{}_{}".format(pid, exercise_type, body_part))
#         count += 1
#     plt.tight_layout()
#     plt.savefig("/tmp/peaks2/{}_{}_all.jpg".format(pid, exercise_type))
#     plt.close()

def plot_body_parts_all(df, pid, exercise_type, final_peaks):
    column_list = df.columns.tolist()
    body_parts_list = ["RElbow_Y", "LElbow_Y", "RWrist_Y", "LWrist_Y"]
    fig = plt.figure(figsize=(20, 10))
    count = 0
    for bp in body_parts_list:
        ax = fig.add_subplot(2, 2, count + 1)
        smooth_coordinates = smooth_coordinates_sf(df[bp])
        index = np.arange(len(smooth_coordinates))
        _ = ax.plot(smooth_coordinates, label='smooth signal')
        _ = ax.plot(index[final_peaks], smooth_coordinates[final_peaks], 'ro', label='minima peaks')
        _ = ax.set_title("{}".format(bp))
        ax.legend()
        count += 1
    fig.suptitle("{}_{}_{}".format(pid, exercise_type, len(final_peaks)), fontsize=14)
    plt.tight_layout()
    plt.savefig("/tmp/peaks4/{}_{}.jpg".format(pid, exercise_type))
    plt.close()


def plot_body_parts(df, pid, exercise_type, body_part, final_peaks):
    smooth_coordinates = smooth_coordinates_sf(df[body_part])
    index = np.arange(len(smooth_coordinates))
    plt.figure(figsize=(20, 5))
    plt.plot(smooth_coordinates)
    plt.plot(index[final_peaks], smooth_coordinates[final_peaks], 'ro', label='minima peaks')
    plt.title("{}_{}_{}".format(pid, exercise_type, len(final_peaks)))
    plt.savefig("/tmp/peaks4/{}_{}.jpg".format(pid, exercise_type))
    plt.close()


def drop_columns(df, important_parts, drop_columns=None):
    try:
        column_list = df.columns.tolist()
        drop_columns_list = list(set(column_list) - set(important_parts))
        if drop_columns:
            drop_columns_list = drop_columns + drop_columns_list
        df = df.drop(drop_columns_list, axis=1)
        return df
    except Exception as e:
        logger.error("Error in dropping the columns: {}".format(str(e)))


def replace_with_mean(df):
    try:
        df = df.replace(0, df.mean())
        df = df.replace(np.nan, df.mean())
        return df
    except Exception as e:
        logger.error("Error in replacing with mean: {}".format(str(e)))


def merge(df_x, df_y):
    try:
        df_total = pd.DataFrame([], columns=df_x.columns)
        for each_column in df_x.columns:
            new_ = np.sqrt(df_x[each_column] * df_x[each_column] + df_y[each_column] * df_y[each_column])
            df_total[each_column] = new_
        return df_total
    except Exception as e:
        logger.error("Error in merging the x and y dataframe: {}".format(str(e)))
        return None


def merge_centre(df_x, df_y):
    try:
        df_total = pd.DataFrame([], columns=df_x.columns)
        mean_x = df_x[['LHip', 'RHip']].mean(axis=1)
        mean_y = df_y[['LHip', 'RHip']].mean(axis=1)

        for each_column in df_x.columns:
            new_ = np.sqrt((df_x[each_column] - mean_x) * (df_x[each_column] - mean_x) +
                           (df_y[each_column] - mean_y) * (df_y[each_column] - mean_y))
            df_total[each_column] = new_
        return df_total
    except Exception as e:
        logger.error("Error in merging (centric) the x and y dataframe: {}".format(str(e)))
        return None


def merge_helper(x_, y_, ignore_x, merge_type):
    df_merged = pd.concat([x_, y_], axis=1)
    return df_merged


def standardize_helper(df_merged, scaling_algo, important_parts):
    column_list = df_merged.columns.tolist()
    if scaling_algo == "znorm":
        for col in column_list:
            if col not in important_parts:
                continue
            df_merged[col] = (df_merged[col] - df_merged[col].mean()) / df_merged[col].std(ddof=0)

    if scaling_algo == "minmax":
        for col in column_list:
            if col not in important_parts:
                continue
            df_merged[col] = (df_merged[col] - df_merged[col].min()) / (df_merged[col].max() - df_merged[col].min())
    return df_merged


def standardize_np_array(combined_data):
    """
    Function to normalize a numpy matrix of matrix, for multivariate time series
    """
    number_records = combined_data.shape[0]
    scaled_combined_data = []
    for i in range(number_records):
        single_record = combined_data[i]
        single_record = (single_record - single_record.mean(axis=0)) / single_record.std(axis=0)
        scaled_combined_data.append(single_record)
    scaled_combined_data = np.array(scaled_combined_data)
    return scaled_combined_data


def create_custom_exercise_features(df):
    df["wrist_lr_shoulder_distance"] = np.sqrt(
        ((df["LWrist"] + df["LShoulder"]) / 2.0 - (df["RWrist"] + df["RShoulder"]) / 2.0) ** 2)
    df["hip_shoulder_distance"] = np.abs((df["LShoulder"] + df["RShoulder"]) / 2.0 - (df["LHip"] + df["RHip"]) / 2.0)
    df["wrist_shoulder_distance"] = np.min(
        (df["LWrist"] + df["RWrist"]) / 2.0 - (df["LShoulder"] + df["RShoulder"]) / 2.0)
    df["wrist_hip_distance"] = np.sqrt(((df["LWrist"] + df["RWrist"]) / 2.0 - (df["LHip"] + df["RHip"]) / 2.0) ** 2)
    return df


def custom_segment(df_merged, peaks_max_y, increment=1):
    sample_id = [-1] * df_merged.shape[0]
    count = 1
    # Use it to equally divide by total repetitions
    # peaks_max_y[-1] -= 1
    for i in range(0, len(peaks_max_y) - 1, increment):
        starting = peaks_max_y[i]
        ending = -1
        if i + increment <= len(peaks_max_y) - 1:
            ending = peaks_max_y[i + increment]

        if ending > starting:
            for j in range(starting, ending):
                sample_id[j] = count
        count += 1
    return sample_id

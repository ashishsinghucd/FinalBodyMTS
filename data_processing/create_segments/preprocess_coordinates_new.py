import argparse
import json
import os
import sys
import configparser
import traceback
from pathlib import Path
import logging
import warnings
import time

warnings.filterwarnings("ignore")

import seaborn as sns
import pandas as pd
import numpy as np
import peakutils
import matplotlib.pyplot as plt
from configobj import ConfigObj

from utils.util_functions import get_unique_list_of_files
from data_processing.create_segments.preprocess_utils import drop_columns, smooth_coordinates_sf, get_peaks, \
    replace_with_mean, merge_helper, custom_segment, standardize_helper, plot_body_parts, plot_body_parts_all
from utils.util_functions import create_directory_if_not_exists, delete_directory_if_exists
from utils.constants import columns_rename_mappingv17

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style('darkgrid')
# Some global variables to generate and store stats of the data
segment_stats = []
ignored_clips_list = []
valid_files_count = 0
frames_info = {}
average_df_list = []
count_zeros = 0
total_shape = 0
total_mean_df = pd.DataFrame()
segment_temp = {"pid": [], "count": []}
exercise_types_mapping = {"MP": ["A", "Arch", "N", "R"], "Rowing": ["A", "Ext", "N", "R", "RB"]}
total_reps = 0

def combine_confidence(prob_):
    """
    :param prob_:
    :return:
    """
    prob_avg_df = prob_.groupby(["PID", "rep"]).mean().reset_index()
    average_df_list.append(prob_avg_df)


def generate_segmented_coordinates(coordinate_base_path, segment_path):
    """
    Function to get the indices of the peaks. It also drops the unnecessary columns, rename the columns and replace
    empty values with the mean value
    """
    stats = {}
    pid = coordinate_base_path.split("_", 1)[0]
    exercise_type = coordinate_base_path.split("_")[1].strip()

    check_file_name = "{}_{}".format(pid, exercise_type)
    if common_pids and check_file_name not in common_pids:
        return
    global valid_files_count
    valid_files_count += 1
    df = pd.read_csv(os.path.join(full_coordinates_path, coordinate_base_path + ".csv"))
    logger.info("Shape of dataframe: {}".format(df.shape))
    x_cols = []
    y_cols = []
    prob_cols = []
    for c in df.columns:
        if c.endswith("_X"):
            x_cols.append(c)
        elif c.endswith("_Y"):
            y_cols.append(c)
        elif c.endswith("_prob"):
            prob_cols.append(c)
        else:
            continue

    x_ = df[x_cols]
    y_ = df[y_cols]

    prob_ = df[prob_cols]
    # x_.columns = x_.columns.str[:-2]
    # y_.columns = y_.columns.str[:-2]
    prob_.columns = prob_.columns.str[:-5]
    prob_.loc[:, "PID"] = ["{}_{}".format(pid, exercise_type)] * prob_.shape[0]

    column_list = x_.columns.tolist()

    # Save a copy of the dataframes in case we do any pre-processing on the original data
    x_copy = x_.copy(deep=True)
    y_copy = y_.copy(deep=True)

    # Basic data pre-processing to handle zero values
    # Replace 0 with the previous non zero value
    # for c in y_copy.columns.tolist():
    #     y_copy[c] = y_copy[c].replace(to_replace=0, method='ffill')
    #
    # for c in x_copy.columns.tolist():
    #     x_copy[c] = x_copy[c].replace(to_replace=0, method='ffill')



    # peak_filtered_parts = list(set(peak_parts) & set(column_list))
    # peaks_max_y = get_peaks(y_, peak_filtered_parts, pid, exercise_type, plot_peaks=True)

    # Get the peaks based on a particular body part, it may vary for different exercises
    # We are using predefined information that is obtained and corrected after using scipy to get the peaks info
    # Uncomment the above lines of code in case we want to generate the fresh peak information
    peaks_max_y = segment_info[pid][exercise_type]

    # We use the below piece of code when we use frame-step of size 2, 3 etc.
    # peaks_max_y = [int(i / 2) for i in peaks_max_y]

    # To collect the peaks stats
    reps_duration_list = [x - peaks_max_y[i - 1] for i, x in enumerate(peaks_max_y) if i > 0]
    # Save the frame information for each person id

    # Plot the body parts for more visualization
    # plot_body_parts(y_, pid, exercise_type, "RElbow_Y", peaks_max_y)
    # plot_body_parts_all(y_, pid, exercise_type, peaks_max_y)
    segment_temp["pid"].append(check_file_name)
    segment_temp["count"].append(len(peaks_max_y) - 1)

    if pid not in frames_info:
        frames_info[pid] = {}
    frames_info[pid][exercise_type] = [int(i) for i in peaks_max_y]

    # TODO check how to get the peaks correctly
    # We know in advance that the number of peaks based on the data cannot be greater than 10 or 11
    if len(peaks_max_y) > max_peaks_time_series:
        ignored_clips_list.append(coordinate_base_path)
        return
    global total_reps
    total_reps += len(peaks_max_y) - 1
    logger.info("Total peaks are: {}".format(len(peaks_max_y)))

    # Merge the x and y coordinates, mark the peaks in the corresponding frame number

    # for c in y_copy.columns.tolist():
    #     y_copy[c] = smooth_coordinates_sf(y_copy[c])

    # We are using both x and y coordinates
    # This function was also meant to combined x and y coordinates into 1 value
    df_merged = merge_helper(x_copy, y_copy, ignore_x, merge_type)

    # We replace 0 values with the neighbors values using the ffill
    # OpenPose outputs 0 in the data when the body part is obstructed or occluded
    for c in df_merged.columns.tolist():
        df_merged[c] = df_merged[c].replace(to_replace=0, method='ffill')

    df_nonzero = df_merged.loc[:, ~df_merged.columns.isin(['Nose_X', 'Nose_Y', "LEar_X", "LEar_Y", "REar_X", "REar_Y",
                                                    "LEye_X", "LEye_Y", "REye_X", "REye_Y", "RBigToe_X", "RSmallToe_X",
                                                           "RBigToe_Y", "RSmallToe_Y", "LBigToe_X", "LSmallToe_X",
                                                           "LBigToe_Y", "LSmallToe_Y"])]

    # This is to get the count of total zeros in the dataframe for more information
    global count_zeros, total_shape
    count_zeros += df_merged[df_merged == 0].count(axis=1).sum()
    total_shape += df_merged.size

    if df_nonzero.eq(0).any().any():
        logger.info("Zero values: {}".format(pid))
        # for c in df_nonzero.columns.tolist():
        #     if (df_nonzero[c] == 0).any():
        #         print(c)

    # Standardize each coordinate with mean 0 and variance 1
    # We don't standardize in case of videos. This is set to False by default. It can be changed later on
    # using in the ROCKET classifier.
    if standardize:
        df_merged = standardize_helper(df_merged, scaling_algo, column_list)

    # We keep this information in the dataframe to segment the time series into individual repetitions later on
    df_merged["frame_number"] = np.arange(df_merged.shape[0])
    df_merged["frame_peaks"] = df_merged["frame_number"].isin(peaks_max_y).astype(int)
    df_merged["pid"] = pid

    # Get the stats
    stats["pid"] = pid
    stats["exercise"] = exercise
    stats["exercise_type"] = exercise_type
    stats["number_of_reps"] = len(peaks_max_y) - 1
    stats["indices_peaks"] = peaks_max_y
    stats["duration_reps"] = reps_duration_list
    stats["average_reps_duration"] = np.round(np.mean(reps_duration_list), 2)
    stats["max_reps_duration"] = np.max(reps_duration_list)
    stats["min_reps_duration"] = np.min(reps_duration_list)
    segment_stats.append(stats)

    # Create a sample id for each segment
    # This information is used to create final segmented time series for each repetition
    sample_id = custom_segment(df_merged, peaks_max_y, increment)
    df_merged["sample_id"] = sample_id
    prob_["rep"] = sample_id
    prob_ = prob_[prob_["rep"] != -1]

    # Merging the confidence
    # The prob_ dataframes store the probability of each body part location. It can be processed for further analysis.
    combine_confidence(prob_)
    # Remove the negative sample ids
    df_merged = df_merged[df_merged["sample_id"] != -1]

    # Save the final dataframe for each participant into a csv file
    df_merged.to_csv(segment_path + "/" + coordinate_base_path + ".csv", index=False)


def smooth_remove_bs(column_list, df):
    """
    Function to remove the baseline signal from a signal. It didn't matter much in our case.
    """
    for col in column_list:
        if df[col].nunique() == 1:
            continue
        smooth_coordinates = smooth_coordinates_sf(df[col])
        baseline = peakutils.baseline(smooth_coordinates, 5)
        scaled = smooth_coordinates - baseline
        df.loc[:, col] = scaled


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_test_config", required=True, help="path of the config file")
    parser.add_argument("--exercise_config", required=True, help="path of the config file")
    args = parser.parse_args()

    # Config file to pass all the parameters
    train_test_config = ConfigObj(args.train_test_config)

    # Read the arguments
    home_path = str(Path.home())
    exercise = train_test_config["EXERCISE"]
    standardize = train_test_config.as_bool("STANDARDIZATION")
    ignore_x = train_test_config.as_bool("IGNORE_X_COORDINATE")
    merge_type = train_test_config["MERGE_TYPE"]
    scaling_algo = train_test_config["SCALING_TYPE"]
    # Read the paths and folders
    segment_stats_dir = train_test_config["SEGMENT_STATS_DIR"]
    segmented_coordinates_dir = train_test_config["SEGMENTED_COORDINATES_DIR"]
    max_peaks_time_series = train_test_config.as_int("MAX_PEAKS_TIME_SERIES")
    increment = train_test_config.as_int("INCREMENT")
    segment_info_file = train_test_config["SEGMENT_INFO_FILE"]
    common_pids = train_test_config["COMMON_PIDS"]

    base_path = os.path.join(home_path, train_test_config["BASE_PATH"], exercise)
    full_coordinates_path = os.path.join(base_path, train_test_config["FULL_COORDINATES_PATH"])

    with open(os.path.join(home_path, segment_info_file)) as f:
        segment_info = json.load(f)

    # full_coordinates_path = os.path.join(full_coordinates_path, exercise)
    full_segmented_coordinates_path = os.path.join(base_path, segmented_coordinates_dir)
    segment_stats_path = os.path.join(base_path, segment_stats_dir)

    delete_directory_if_exists(full_segmented_coordinates_path)
    create_directory_if_not_exists(full_segmented_coordinates_path)
    create_directory_if_not_exists(segment_stats_path)

    config_parser = configparser.RawConfigParser()
    config_parser.read(args.exercise_config)
    important_parts = config_parser.get(exercise, "important_parts").split(",")
    peak_parts = config_parser.get(exercise, "peak_parts").split(",")
    valid_classes = config_parser.get(exercise, "valid_classes").split(",")

    unique_coordinates_files = get_unique_list_of_files(full_coordinates_path, 4)
    if not unique_coordinates_files:
        logger.info("No coordinates files found")
        sys.exit(0)

    generate_segmented_args = []

    logger.info("Total number of coordinates files: {}".format(len(unique_coordinates_files)))

    ts = time.time()
    for coordinate_base_path in unique_coordinates_files:
        logger.info("Running for {}".format(coordinate_base_path))
        try:
            if not coordinate_base_path.startswith('.'):
                generate_segmented_args.append((coordinate_base_path, full_segmented_coordinates_path))
                generate_segmented_coordinates(coordinate_base_path, full_segmented_coordinates_path)
        except Exception as e:
            logger.info("Error in generating the coordinates for: {} {}".format(coordinate_base_path, str(e)))
            logger.info(traceback.format_exc())
    te = time.time()
    total_time = (te - ts)
    logger.info('Total time preprocessing: {} seconds'.format(total_time))

    segment_stats_df = pd.DataFrame(segment_stats)

    stats_col_order = ["pid", "exercise", "exercise_type", "number_of_reps", "average_reps_duration",
                       "max_reps_duration", "min_reps_duration", "duration_reps", "indices_peaks"]
    segment_stats_ordered_df = segment_stats_df[stats_col_order]
    reps_list = segment_stats_ordered_df["duration_reps"].tolist()
    duration_list = []
    for sublist in reps_list:
        for duration in sublist:
            duration_list.append(duration)

    segment_stats_ordered_df = segment_stats_ordered_df[segment_stats_ordered_df["exercise_type"].isin(valid_classes)]

    total_person = len(segment_stats_ordered_df["pid"].unique())
    total_exercise_type = len(segment_stats_ordered_df["exercise_type"].unique())
    total_samples = segment_stats_ordered_df["number_of_reps"].sum()
    overall_average = round(segment_stats_ordered_df["average_reps_duration"].mean(), 2)
    overall_max = segment_stats_ordered_df["max_reps_duration"].max()
    overall_min = segment_stats_ordered_df["min_reps_duration"].min()
    overall_stats = ["total_person: " + str(total_person),
                     None,
                     "total_exercise_type: " + str(total_exercise_type),
                     "total_samples: " + str(total_samples),
                     "Average duration of rep:" + str(overall_average),
                     "Maximum duration of rep:" + str(overall_max),
                     "Minimum duration of rep:" + str(overall_min),
                     None,
                     None]

    segment_stats_ordered_df = segment_stats_ordered_df.sort_values(by=['pid'])
    segment_stats_ordered_df.index.name = "row_number"
    segment_stats_ordered_df.loc['data_info'] = overall_stats
    file_name = "segment_stats"
    current_directory = os.path.join(os.getcwd(), 'stats')
    plt.figure(figsize=(10, 3))
    sns.distplot(duration_list)
    plt.title('Histogram for the duration of reps')
    plt.xlabel('Counts')
    plt.ylabel('Duration')
    plt.savefig(segment_stats_path + "/duration_hist.png")
    segment_stats_ordered_df[stats_col_order].to_csv(segment_stats_path + '/{}.csv'.format(file_name), index=True)
    logger.info("Total number of repetitions for {} are : {}".format(exercise, total_reps))
    logger.info("Total number of coordinates files: {}".format(len(unique_coordinates_files)))
    logger.info("Total valid files are: {}".format(valid_files_count))
    logger.info("Total ignored files are: {}".format(len(ignored_clips_list)))
    logger.info("Total proportion of zeros: {}".format(float(count_zeros) * 100 / total_shape))

    average_df = pd.concat(average_df_list, axis=0)
    average_df.to_csv(segment_stats_path + '/{}.csv'.format(train_test_config["FULL_COORDINATES_PATH"]), index=False)

    segment_temp_df = pd.DataFrame(segment_temp)
    segment_temp_df.to_csv("/tmp/hpe_shimmer.csv", index=False)

    # save the frame information again in case of getting peak information again using scipy
    # with open(segment_stats_path + "/frames_info.json", "w") as f:
    #     json.dump(frames_info, f, sort_keys=True, indent=4)

"""

conda config --set auto_activate_base false
conda create -n yourenvname python=x.x
conda remove -n env_name --all

virtualenv pytorch -p /usr/local/bin/python3
"""

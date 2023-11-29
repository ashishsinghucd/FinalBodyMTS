import collections
import configparser
import json
import math
import os
import sys
import traceback
import argparse
import logging
import time
from collections import Counter

import pandas as pd
import numpy as np
from configobj import ConfigObj
from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit

from data_processing.create_train_test_data.create_sktime_format import create_and_save_sktime_format
from data_processing.create_train_test_data.create_train_test_utils import interpolate_coordinates, get_func_length
from data_processing.create_train_test_data.create_tslearn_format import create_and_save_tslearn_format
from data_processing.create_segments.preprocess_utils import standardize_np_array
from utils.math_funtions import get_combinations
from utils.util_functions import create_directory_if_not_exists, delete_directory_if_exists

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
SKTIME_FORMAT_DIR = "TrainTestDataSktime"
TSLEARN_FORMAT_DIR = "TrainTestDataTslearn"

FILE_NAME_X = '{}_{}_X'
FILE_NAME_Y = '{}_{}_Y'
FILE_NAME_PID = '{}_{}_pid'

IGNORE_LEN_COUNT = 0

mapping_dict = {"N": "Normal"}
map_exercise_type = lambda x: mapping_dict.get(x, "Wrong")
exercise_types_mapping = {"MP": ["A", "Arch", "N", "R"], "Rowing": ["A", "Ext", "N", "R", "RB"]}


def train_test_split(segment_df_list, segment_label_list, segment_file_name_list, current_pids_list, gender_info_list,
                     seed_value):
    """
    Function to split the pids in training, testing and validation data
    """
    train_pids, test_pids, train_gender, test_gender = split_pids(current_pids_list, gender_info_list, seed_value,
                                                                        train_test_split_ratio)
    val_pids = np.array([])
    val_gender = np.array([])
    if validation_data:
        train_pids, val_pids, train_gender, val_gender = split_pids(train_pids, train_gender, seed_value,
                                                                          split_val_ratio)
    logger.info("Full data gender distribution are: {}".format(str(Counter(gender_info_list))))

    logger.info("Training pids are: {}".format(str(train_pids)))
    logger.info("Train data gender distribution are: {}".format(str(Counter(train_gender))))
    logger.info("Testing pids are: {}".format(str(test_pids)))
    logger.info("Test data gender distribution are: {}".format(str(Counter(test_gender))))

    logger.info("Validation pids are: {}".format(str(val_pids)))
    logger.info("Validation data gender distribution are: {}".format(str(Counter(val_gender))))


    pid_dict["total"] = {}
    pid_dict["total"]["training_pid"] = list(train_pids)
    pid_dict["total"]["testing_pid"] = list(test_pids)
    pid_dict["total"]["val_pids"] = list(val_pids)

    train_segments_x, train_label_y, train_pids_order, train_file_name_x = [], [], [], []
    val_segments_x, val_label_y, val_pids_order, test_file_name_x = [], [], [], []
    test_segments_x, test_label_y, test_pids_order, val_file_name_x = [], [], [], []

    for i, segment_df in enumerate(segment_df_list):
        try:
            pid = segment_df["pid"].iloc[0]
            segment_df = segment_df.drop(["frame_number", "frame_peaks", "sample_id", "pid"], axis=1)
            if pid in train_pids:
                train_segments_x.append(segment_df)
                train_label_y.append(segment_label_list[i])
                train_pids_order.append(pid)
                train_file_name_x.append(segment_file_name_list[i])
            elif pid in test_pids:
                test_segments_x.append(segment_df)
                test_label_y.append(segment_label_list[i])
                test_pids_order.append(pid)
                test_file_name_x.append(segment_file_name_list[i])
            else:
                val_segments_x.append(segment_df)
                val_label_y.append(segment_label_list[i])
                val_pids_order.append(pid)
                val_file_name_x.append(segment_file_name_list[i])
        except Exception as e:
            logger.info("Error creating the train test data: {}".format(str(e)))
            logger.info(segment_df.shape)
    logger.info("Total segments are: {}".format(len(segment_df_list)))
    logger.info("Training segments: {}, Testing segments: {} Validation segments: {}"
                .format(len(train_segments_x), len(test_segments_x), len(val_segments_x)))
    train_label_y = np.array(train_label_y)
    test_label_y = np.array(test_label_y)
    val_label_y = np.array(val_label_y)

    train_pids_order = np.array(train_pids_order)
    test_pids_order = np.array(test_pids_order)
    val_pids_order = np.array(val_pids_order)

    train_file_name_x = np.array(train_file_name_x)
    test_file_name_x = np.array(test_file_name_x)
    val_file_name_x = np.array(val_file_name_x)

    # To shuffle the records in the training, testing and validation

    # perm1 = np.random.permutation(len(train_segments_x))
    # perm2 = np.random.permutation(len(test_segments_x))
    # perm3 = np.random.permutation(len(val_segments_x))
    # train_segments_x, train_label_y, train_file_name_x = [train_segments_x[i] for i in perm1], train_label_y[perm1], \
    #                                                      train_file_name_x[perm1]
    # test_segments_x, test_label_y, test_file_name_x = [test_segments_x[i] for i in perm2], test_label_y[perm2], \
    #                                                   test_file_name_x[perm2]
    # val_segments_x, val_label_y, val_file_name_x = [val_segments_x[i] for i in perm3], val_label_y[perm3], \
    #                                                val_file_name_x[perm3]

    return train_segments_x, train_label_y, test_segments_x, test_label_y, val_segments_x, val_label_y, \
           train_file_name_x, test_file_name_x, val_file_name_x


def split_pids(current_pids_list, gender_list, seed_value, split_ratio):
    """
    Function to split the pids into two arrays
    """
    list_pids = current_pids_list
    list_pids.sort()
    logger.info("Current pids are: {}".format(str(current_pids_list)))
    person_count_test = math.floor(split_ratio * len(list_pids))
    person_count_train = len(list_pids) - person_count_test
    logger.info("Total persons in training: {}, testing/validation: {}".format(person_count_train, person_count_test))
    seed_value = int(seed_value)
    np.random.seed(seed_value)
    train_pids = np.random.choice(list_pids, person_count_train, replace=False)
    test_pids = np.array(list(set(list_pids) - set(train_pids)))
    return train_pids, test_pids, np.array([]), np.array([])


def split_pids_label(current_pids_list, gender_list, seed_value, split_ratio):
    """
    Function to split the pids into two arrays
    """
    list_pids = current_pids_list
    list_pids.sort()
    logger.info("Current pids are: {}".format(str(current_pids_list)))
    seed_value = int(seed_value)
    current_pids_list = np.array(current_pids_list)
    gender_list = np.array(gender_list)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=split_ratio, random_state=seed_value)
    for train_index, test_index in sss.split(current_pids_list, gender_list):
        train_pids, test_pids = current_pids_list[train_index], current_pids_list[test_index]
        train_gender, test_gender = gender_list[train_index], gender_list[test_index]

    logger.info("Total persons in training: {}, testing/validation: {}".format(len(train_pids), len(test_pids)))
    # seed_value = int(seed_value)
    # np.random.seed(seed_value)
    # train_pids = np.random.choice(list_pids, person_count_train, replace=False)
    # test_pids = np.array(list(set(list_pids) - set(train_pids)))
    return train_pids, test_pids, train_gender, test_gender


def create_final_formatted_data(seed_value):
    clip_files_list = os.listdir(full_segmented_coordinates_path)
    clip_files_list = [f for f in clip_files_list if not f.startswith(".")]
    clip_df_list0 = []
    clip_label_list = []
    clip_name_list = []
    logger.info("Total clips are:{}".format(len(clip_files_list)))
    # First read all the clips using the pandas and store all dataframes in a list
    # Also store the name of the file to get pid info and exercise type
    for clip_file in clip_files_list:
        try:
            pid = clip_file.split("_")[0]
            file_name = clip_file.rsplit(".", 1)[0]
            exercise_type = clip_file.split("_")[1].strip()[:-4]
            # pid = clip_file.split(" ")[0]
            # file_name = clip_file.rsplit(".", 1)[0]
            # exercise_type = clip_file.split("_")[3].strip()
            if exercise_type not in valid_classes:
                continue
            clip_df_list0.append(pd.read_csv(os.path.join(full_segmented_coordinates_path, clip_file)))
            clip_label_list.append(exercise_type)
            clip_name_list.append(file_name)
        except FileNotFoundError as e:
            logger.info(str(e))

    clip_df_list = []
    # Drop unnecessary parts
    for df in clip_df_list0:
        df.drop(drop_parts, axis=1, inplace=True)
        clip_df_list.append(df)

    # Have the same order of columns in all the dataframes
    column_list = clip_df_list[0].columns.tolist()
    logger.info("Column order: {}".format(column_list))
    clip_df_list = [df[column_list] for df in clip_df_list]

    current_pids_list = []
    segment_df_list = []
    segment_label_list = []
    segment_file_name_list = []
    # Separate each dataframe for each repetition
    for i, clip_df in enumerate(clip_df_list):
        unique_sample_id_list = clip_df["sample_id"].unique()
        for sample_id in unique_sample_id_list:
            segment_df = clip_df[clip_df["sample_id"] == sample_id].copy()
            if min_length_time_series != -1 and segment_df.shape[0] < min_length_time_series:
                print(segment_df["pid"].iloc[0], segment_df.shape[0])
                global IGNORE_LEN_COUNT
                IGNORE_LEN_COUNT += 1
                continue

            starting_frame = segment_df["frame_number"].min()
            ending_frame = segment_df["frame_number"].max()
            pid = segment_df["pid"].iloc[0]
            sample_id = segment_df["sample_id"].iloc[0]
            current_pids_list.append(pid)
            # Standardize each coordinate
            # if standardize:
            #     segment_df = standardize_helper(segment_df, scaling_algo, important_parts)
            segment_df_list.append(segment_df)
            segment_label_list.append(clip_label_list[i])
            segment_file_name_list.append((clip_name_list[i], starting_frame, ending_frame, sample_id))

    label_clip_count = collections.Counter(clip_label_list)
    label_segment_count = collections.Counter(segment_label_list)

    current_pids_list = list(set(current_pids_list))
    gender_info_list = []
    for i in current_pids_list:
        gender_info_list.append(gender_info[i])
    result = train_test_split(segment_df_list, segment_label_list, segment_file_name_list, current_pids_list,
                              gender_info_list, seed_value)
    train_segments_x, train_label_y, test_segments_x, test_label_y, = result[0], result[1], result[2], result[3]
    val_segments_x, val_label_y = result[4], result[5]
    train_file_name_x, test_file_name_x, val_file_name_x = result[6], result[7], result[8]
    # Info of the exercise types
    train_segment_count = collections.Counter(train_label_y)
    test_segment_count = collections.Counter(test_label_y)
    val_segment_count = collections.Counter(val_label_y)

    for k in valid_classes:
        split_stats.append({"exercise_type": k,
                            "clip_count": label_clip_count[k],
                            "segment_count": label_segment_count.get(k, 0),
                            "train_segment_count": train_segment_count.get(k, 0),
                            "test_segment_count": test_segment_count.get(k, 0),
                            "val_segment_count": val_segment_count.get(k, 0)
                            })

    train_array_list = [df.values for df in train_segments_x]
    test_array_list = [df.values for df in test_segments_x]
    val_array_list = [df.values for df in val_segments_x]

    combined_train = np.array(train_array_list, dtype=object)
    combined_test = np.array(test_array_list, dtype=object)
    combined_val = np.array(val_array_list, dtype=object)

    combined_train = np.array([np.flipud(i) for i in combined_train], dtype=object)
    combined_test = np.array([np.flipud(i) for i in combined_test], dtype=object)
    combined_val = np.array([np.flipud(i) for i in combined_val], dtype=object)

    # In case we also want to reverse the clip
    if reverse_clip:
        combined_train_reverse = np.array([np.flipud(i) for i in combined_train], dtype=object)
        # combined_test_reverse = np.array([np.flipud(i) for i in combined_test], dtype=object)
        # combined_val_reverse = np.array([np.flipud(i) for i in combined_val], dtype=object)
        combined_train = combined_train_reverse
        # combined_train = np.hstack([combined_train, combined_train_reverse])
        # combined_test = np.hstack([combined_test, combined_test_reverse])
        # combined_val = np.hstack([combined_val, combined_val_reverse])

        # train_label_y = np.hstack([train_label_y, train_label_y])
        # test_label_y = np.hstack([test_label_y, test_label_y])
        # val_label_y = np.hstack([val_label_y, val_label_y])

    logger.info("Full data sizes before saving")
    if len(combined_train):
        logger.info("Training data shape: {} First training record shape: {}".format(combined_train.shape,
                                                                                     combined_train[0].shape))
    if len(combined_test):
        logger.info(
            "Testing data shape: {} First testing record shape: {}".format(combined_test.shape, combined_test[0].shape))
    if len(combined_val):
        logger.info("Validation data shape: {} First validation record shape: {}".format(combined_val.shape,
                                                                                         combined_val[0].shape))

    max_length = max_length_default
    if padding:
        max_length = max(get_func_length(combined_train, func=max, second_max=False),
                         get_func_length(combined_test, func=max), get_func_length(combined_val, func=max))

    try:
        combined_data_list = [(combined_train, train_label_y, "TRAIN", train_file_name_x), (combined_test, test_label_y,
                                                                                            "TEST", test_file_name_x),
                              (combined_val, val_label_y, "VAL", val_file_name_x)]
        for combined_data_tuple in combined_data_list:
            save_pairwise_data(combined_data_tuple, max_length)
        save_library_specific_format()
    except Exception as e:
        logger.info("Error saving the full dataset: {} {}".format(train_test_dir_path, str(e)))
        logger.info(traceback.format_exc())


def save_pairwise_data(combined_data_tuple, max_length):
    combined_data, labels, name, pids_order = combined_data_tuple
    logger.info("Creating train/test for: " + multiclass_combination)
    filtered_combined_data = combined_data
    filtered_label = labels
    filtered_pids_order = pids_order

    if do_interpolate:
        logger.info("Interpolating the data")
        if len(filtered_combined_data):
            filtered_combined_data = interpolate_coordinates(filtered_combined_data, max_length, padding)

    if standardize:
        filtered_combined_data = standardize_np_array(filtered_combined_data)

    output_path = os.path.join(train_test_dir_path, multiclass_combination)
    create_directory_if_not_exists(output_path)

    # Save the data in the numpy format
    np.save(os.path.join(output_path, FILE_NAME_X.format(name, data_type) + ".npy"), filtered_combined_data)
    np.save(os.path.join(output_path, FILE_NAME_Y.format(name, data_type) + ".npy"), filtered_label)
    np.save(os.path.join(output_path, FILE_NAME_PID.format(name, data_type) + ".npy"), filtered_pids_order)


def save_library_specific_format():
    """
    Function to save the tslearn and sktime format from a numpy format
    """
    combination = multiclass_combination
    train_test_combination_path = os.path.join(train_test_dir_path, combination)
    sktime_format_path = os.path.join(base_path, SKTIME_FORMAT_DIR, seed_value, combination)
    # delete_directory_if_exists(sktime_format_path)
    create_directory_if_not_exists(sktime_format_path)
    data_types_tuple = ["TRAIN", "TEST", "VAL"]
    for data_name in data_types_tuple:
        data_name_x = FILE_NAME_X.format(data_name, data_type)
        data_name_y = FILE_NAME_Y.format(data_name, data_type)
        data_pid = FILE_NAME_PID.format(data_name, data_type)
        logger.info("Creating the Sktime format for {} data".format(data_name_x))
        create_and_save_sktime_format(train_test_combination_path, sktime_format_path,
                                      data_name_x, data_name_y, data_pid)

    """
    Commenting out as of now, don't need this format
    tslearn_format_path = os.path.join(base_path, TSLEARN_FORMAT_DIR, exercise, seed_value, combination)
    delete_directory_if_exists(tslearn_format_path)
    create_directory_if_not_exists(tslearn_format_path)
    for data_name_x, data_name_y, data_pid in data_types_tuple:
        logger.info("Creating the tslearn format for {} data".format(data_name_x))
        create_and_save_tslearn_format(train_test_combination_path, tslearn_format_path,
                                       data_name_x, data_name_y, data_pid)
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_test_config", required=True, help="path of the config file")
    parser.add_argument("--exercise_config", required=True, help="path of the config file")
    args = parser.parse_args()

    train_test_config = ConfigObj(args.train_test_config)

    # Read the arguments
    home_path = str(Path.home())
    exercise = train_test_config["EXERCISE"]
    do_interpolate = train_test_config.as_bool("INTERPOLATION")
    padding = train_test_config.as_bool("PADDING")
    input_segmented_coordinates_path = train_test_config["INPUT_DATA_PATH"]
    train_test_split_ratio = train_test_config.as_float("SPLIT_RATIO")
    seed_values = train_test_config.as_list("SEED_VALUES")

    validation_data = train_test_config.as_bool("VALIDATION_DATA")
    split_val_ratio = train_test_config.as_float("SPLIT_VAL_RATIO")

    segment_stats_dir = train_test_config["SEGMENT_STATS_DIR"]
    train_test_dir = train_test_config["TRAIN_TEST_DIR"]
    multiclass_combination = train_test_config["MULTICLASS_DIR"]
    min_length_time_series = train_test_config.as_int("MIN_LENGTH_TIME_SERIES")
    no_of_classes = train_test_config.as_int("NO_OF_CLASSES")
    standardize = train_test_config.as_bool("STANDARDIZATION")
    reverse_clip = train_test_config.as_bool("REVERSE_CLIP")
    scaling_algo = train_test_config["SCALING_TYPE"]
    gender_info_file = train_test_config["GENDER_INFO"]
    # TODO hard coded the value of max length
    max_length_default = int(train_test_config["MAX_LENGTH"])
    train_percentage = int(train_test_split_ratio * 100)
    test_percentage = 100 - train_percentage
    drop_parts = train_test_config.as_list("DROP_PARTS")
    data_type = train_test_config["DATA_TYPE"]
    drop_parts = list(filter(None, drop_parts))
    # train_test_dir = train_test_dir + "_" + str(train_percentage) + "_" + str(test_percentage)

    config_parser = configparser.RawConfigParser()
    config_parser.read(args.exercise_config)
    valid_classes = config_parser.get(exercise, "valid_classes").split(",")
    class_combination = config_parser.get(exercise, "class_combination")
    important_parts = config_parser.get(exercise, "important_parts")
    if not class_combination:
        class_combination = get_combinations(valid_classes, no_of_classes)
    else:
        class_combination = class_combination.split(",")
    label_index_mapping = {i + 1: value for i, value in enumerate(valid_classes)}
    index_label_mapping = {value: i + 1 for i, value in enumerate(valid_classes)}

    base_path = os.path.join(home_path, train_test_config["BASE_PATH"], exercise)

    gender_info_df = pd.read_csv(gender_info_file, sep=",")
    gender_info = dict(zip(gender_info_df.Participant, gender_info_df.Sex))

    # For each seed value create train/test data
    for i, seed_value in enumerate(seed_values):
        split_stats = []
        pid_dict = {}
        logger.info("------------------------------------------------------")
        logger.info("Creating train/test split for seed {}".format(seed_value))
        full_segmented_coordinates_path = os.path.join(base_path, input_segmented_coordinates_path)
        train_test_dir_path = os.path.join(base_path, train_test_dir, seed_value)

        # delete_directory_if_exists(train_test_dir_path)
        create_directory_if_not_exists(train_test_dir_path)
        try:
            ts = time.time()
            create_final_formatted_data(seed_value)
            te = time.time()
            total_time = (te - ts)
            logger.info('Total time preprocessing: {} seconds'.format(total_time))
        except Exception as e:
            logger.info("Error generating the full dataset: {} {}".format(train_test_dir, str(e)))
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

        logger.info("Count of ignore files because of length: {}".format(IGNORE_LEN_COUNT / len(seed_values)))

        logger.info("------------------------------------------------------")
        segment_stats_path = os.path.join(base_path, segment_stats_dir, seed_value)
        create_directory_if_not_exists(segment_stats_path)
        # Creating stats of the data
        split_stats_df = pd.DataFrame(split_stats)
        split_stats_df = split_stats_df.set_index("exercise_type")

        split_col_order = ["clip_count", "segment_count", "train_segment_count", "test_segment_count"]
        split_stats_df = split_stats_df[split_col_order]

        for combination in class_combination:
            classes_list = combination.split("vs")
            split_stats_df["{}_train".format(combination)] = np.where(split_stats_df.index.isin(classes_list),
                                                                      split_stats_df["train_segment_count"], 0)
            split_stats_df["{}_test".format(combination)] = np.where(split_stats_df.index.isin(classes_list),
                                                                     split_stats_df["test_segment_count"], 0)

        split_stats_df.loc['Total', :] = split_stats_df.sum(axis=0)

        split_stats_df.to_csv(segment_stats_path + '/split_stats.csv', index=True)

        with open(segment_stats_path + '/pids_info.json', 'w') as fp:
            json.dump(pid_dict, fp)


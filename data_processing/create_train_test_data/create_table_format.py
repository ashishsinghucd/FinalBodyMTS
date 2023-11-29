import configparser
import os
import sys

import numpy as np
import pandas as pd

from data_processing.create_segments import TRAIN_DATASET_X, TEST_DATASET_X, TRAIN_DATASET_Y, TEST_DATASET_Y
from utils.math_funtions import get_combinations
from utils.util_functions import create_directory_if_not_exists, delete_directory_if_exists


def create_and_save_table_format(dataset_type_x, dataset_type_y, file_extension):
    x_array = np.load(train_test_dir_path + '/{}.npy'.format(dataset_type_x), allow_pickle=True)
    label_array = np.load(train_test_dir_path + '/{}.npy'.format(dataset_type_y), allow_pickle=True)
    file_name = os.path.join(table_format_path, "{}".format(file_extension))
    print("Running the table format.....")
    try:
        x_new_array = []
        for i, oldarray in enumerate(x_array):
            # oldarray = oldarray[:, :-1]
            newarray = np.zeros((oldarray.shape[0], oldarray.shape[1] + 3))
            newarray[:, 3:] = oldarray
            newarray[:, 0] = i + 1
            newarray[:, 1] = np.arange(1, oldarray.shape[0] + 1)
            newarray[:, 2] = label_array[i]
            x_new_array.append(newarray)
        x_new_array = [pd.DataFrame(d) for d in x_new_array]
        for df in x_new_array:
            for j in range(3):
                df[j] = df[j].astype(int)
        full_df = pd.concat(x_new_array)
        full_df.to_csv(file_name, header=None, sep=' ', index=False)
    except Exception as e:
        print("Error writing the table format: {}".format(str(e)))


if __name__ == "__main__":
    base_path = sys.argv[1]
    exercise = sys.argv[2]
    train_test_dir = sys.argv[3]
    multiclass_dir = "MulticlassSplit"
    ratio = train_test_dir.split("_", 1)[1]
    table_format_dir = "TrainTestDataTable" + "_" + ratio

    configParser = configparser.RawConfigParser()
    configFilePath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs/exercise_config')
    configParser.read(configFilePath)
    valid_classes = configParser.get(exercise, "valid_classes").split(",")
    class_combination = configParser.get(exercise, "class_combination")
    if not class_combination:
        class_combination = get_combinations(valid_classes, 2)
    else:
        class_combination = class_combination.split(",")
    label_index_mapping = {i + 1: value for i, value in enumerate(valid_classes)}
    index_label_mapping = {value: i + 1 for i, value in enumerate(valid_classes)}

    all_class_combination = class_combination + [multiclass_dir]
    for combination in all_class_combination:
        train_test_dir_path = os.path.join(base_path, train_test_dir, exercise, combination)
        if not os.path.exists(train_test_dir_path):
            print("Path does not exist: {}".format(train_test_dir_path))
            continue
        classes_list = combination.split("vs")
        if combination == multiclass_dir:
            classes_list = valid_classes
        table_format_path = os.path.join(base_path, table_format_dir, exercise, combination)
        delete_directory_if_exists(table_format_path)
        create_directory_if_not_exists(table_format_path)
        print("Creating the table format for Training data.....{}".format(combination))
        create_and_save_table_format(TRAIN_DATASET_X, TRAIN_DATASET_Y, file_extension=combination + "_TRAIN3")
        print("Creating the table format for Testing data.....{}".format(combination))
        create_and_save_table_format(TEST_DATASET_X, TEST_DATASET_Y, file_extension=combination + "_TEST3")

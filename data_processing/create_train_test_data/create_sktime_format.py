import configparser
import os
import sys
import logging

import numpy as np

from data_processing.create_segments import TRAIN_DATASET_X, TEST_DATASET_X, TRAIN_DATASET_Y, TEST_DATASET_Y
from utils.math_funtions import get_combinations
from utils.util_functions import create_directory_if_not_exists, delete_directory_if_exists

DATA_DESCRIPTION = "#The data consists of coordinates for different body parts which are generated using\n " \
                   "#human pose estimation library. The total participants are 54 and total exercise types are 4\n"
PROBLEM_NAME = "HumanPoseEstimation"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_and_save_sktime_format(train_test_dir_path, sktime_format_path, dataset_type_x,
                                  dataset_type_y, data_pid):
    # The order of data in the numpy array is same as in the sktime format because we are iterating sequentially
    x_array = np.load(train_test_dir_path + '/{}.npy'.format(dataset_type_x), allow_pickle=True)
    label_array = np.load(train_test_dir_path + '/{}.npy'.format(dataset_type_y), allow_pickle=True)
    pid_array = np.load(train_test_dir_path + '/{}.npy'.format(data_pid), allow_pickle=True)

    if not len(x_array):
        logger.info("Skipping sktime format. Data is empty")
        return

    classes_list = np.unique(label_array)
    all_labels = ",".join(classes_list)

    file_name = os.path.join(sktime_format_path, "{}.ts".format(dataset_type_x))
    logger.info("Running the Sktime format.....")
    try:
        with open(file_name, 'w') as f:
            f.write("@problemName '{problem_name}'\n".format(problem_name=PROBLEM_NAME))
            f.write("@timeStamps false\n")
            f.write("\n")
            f.write("\n")
            f.write("@classLabel true {}\n".format(all_labels))
            f.write("@univariate false\n")
            f.write("@data\n")
            l = len(x_array)
            for i, record in enumerate(x_array):
                # record = record[:, :-1]
                record_transpose = record.T
                np.savetxt(f, record_transpose, delimiter=",", fmt='%f', newline=":")
                label = label_array[i]  # label_index_mapping[int(label_array[i])]
                if i == l - 1:
                    f.write("{}".format(label))
                    continue
                f.write("{}\n".format(label))
                # if i and not (i % 50):
                #     print("Written {} records".format(str(i)))
        np.save(sktime_format_path + '/{}.npy'.format(data_pid), pid_array)
    except Exception as e:
        logger.info("Error writing the Sktime format: {}".format(str(e)))

    logger.info("Finished writing the Sktime format")


if __name__ == "__main__":
    base_path = sys.argv[1]
    exercise = sys.argv[2]
    train_test_dir = sys.argv[3]
    multiclass_dir = "MulticlassSplit"
    sktime_format_dir = "TrainTestDataSktime"

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

        sktime_format_path = os.path.join(base_path, sktime_format_dir, exercise, combination)
        delete_directory_if_exists(sktime_format_path)
        create_directory_if_not_exists(sktime_format_path)
        logger.info("Creating the Sktime format for Training data.....{}".format(combination))
        create_and_save_sktime_format(train_test_dir_path, classes_list, sktime_format_path, TRAIN_DATASET_X, TRAIN_DATASET_Y)
        logger.info("Creating the Sktime format for Testing data.....{}".format(combination))
        create_and_save_sktime_format(train_test_dir_path, classes_list, sktime_format_path, TEST_DATASET_X, TEST_DATASET_Y)
    # print("Creating the Sktime format for testing data using all pairs.....")
    # train_test_dir_path = os.path.join(base_path, train_test_dir, exercise, "CombinedTest")
    # classes_list = valid_classes
    # sktime_format_path = os.path.join(base_path, sktime_format_dir, exercise, "CombinedTest")
    # delete_directory_if_exists(sktime_format_path)
    # create_directory_if_not_exists(sktime_format_path)
    # create_and_save_sktime_format(classes_list, TEST_DATASET_X, TEST_DATASET_Y)


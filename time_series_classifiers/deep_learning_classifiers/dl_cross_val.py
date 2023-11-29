import argparse
import configparser
import os
from pathlib import Path
import logging
import getpass

import numpy as np
import pandas as pd
from configobj import ConfigObj
from sklearn.preprocessing import OneHotEncoder

from data_processing.extract_segments import TRAIN_DATASET_Y, TRAIN_DATASET_X, TEST_DATASET_X, TEST_DATASET_Y
from time_series_classifiers.deep_learning_classifiers.run_dl import fit_classifier, create_classifier
from utils.util_functions import create_directory_if_not_exists

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
dataset_name = "MulticlassSplit"


def fit_classifier(classifier_name, datasets_dict, dataset_name):
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    y_train_orig = y_train
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    classifier = create_classifier(classifier_name, input_shape, nb_classes, weights_directory, exercise)
    # with tf.device('/gpu:1'):
    classifier.fit(x_train, y_train, x_test, y_test, y_true, nb_epochs)
    confusion_matrix, classification_report, df_metrics = classifier.predict(x_test, y_true, enc)
    accuracy = df_metrics.loc[0, "accuracy"]
    return accuracy


def generate_cross_val_splits(unique_pids, pid_full_df):
    for pid in unique_pids:
        logger.info("Running for pid : {}".format(pid))
        test_ix = np.where(pid_full_df["pid"] == pid)[0]
        train_ix = np.argwhere(~pid_full_df.index.isin(test_ix)).reshape(-1)
        yield train_ix, test_ix


def loocv_dl(x_full_dataset, y_full_dataset, unique_pids, pid_full_df, exercise):
    scores = []
    datasets_dict = {}
    for train_ix, test_ix in generate_cross_val_splits(unique_pids, pid_full_df):
        x_train_pid, x_test_pid = x_full_dataset[train_ix, :], x_full_dataset[test_ix, :]
        y_train_pid, y_test_pid = y_full_dataset[train_ix], y_full_dataset[test_ix]
        datasets_dict[dataset_name] = (x_train_pid.copy(), y_train_pid.copy(), x_test_pid.copy(), y_test_pid.copy())
        logger.info("Training shape: {}".format(x_train_pid.shape))
        logger.info("Testing shape: {}".format(x_test_pid.shape))

        accuracy = fit_classifier(classifier_type, datasets_dict, dataset_name)
        scores.append(accuracy)
        logger.info("The accuracy: {} ".format(accuracy))
    logger.info("The average accuracy: {} std: {}".format(np.mean(scores), np.std(scores)))


def combine_data(data_path):
    x_train = np.load(os.path.join(data_path, "{}.npy".format(TRAIN_DATASET_X)), allow_pickle=True)
    y_train = np.load(os.path.join(data_path, "{}.npy".format(TRAIN_DATASET_Y)), allow_pickle=True)
    x_test = np.load(os.path.join(data_path, "{}.npy".format(TEST_DATASET_X)), allow_pickle=True)
    y_test = np.load(os.path.join(data_path, "{}.npy".format(TEST_DATASET_Y)), allow_pickle=True)
    pid_train = np.load(os.path.join(data_path, "TRAIN_pid.npy".format(TRAIN_DATASET_X)), allow_pickle=True)
    pid_test = np.load(os.path.join(data_path, "TEST_pid.npy".format(TRAIN_DATASET_X)), allow_pickle=True)
    x_full_dataset = np.concatenate([x_train, x_test], axis=0)
    y_full_dataset = np.concatenate([y_train, y_test], axis=0)
    pid_full = np.concatenate([pid_train, pid_test], axis=0)
    unique_pids = np.unique([i.split(" ")[0] for i in pid_full[:, 0]])
    pid_full_df = pd.DataFrame(pid_full, columns=["file_name", "start_time", "end_time"])
    pid_full_df["pid"] = pid_full_df["file_name"].apply(lambda x: x.split(" ")[0])

    return x_full_dataset, y_full_dataset, unique_pids, pid_full_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dl_config", required=True, help="path of the config file")
    args = parser.parse_args()
    dl_config = ConfigObj(args.dl_config)

    use_engine = False
    use_sonic = False

    if getpass.getuser() == "ashish":
        use_engine = True
    elif getpass.getuser() == "19205522":
        use_sonic = True

    home_path = str(Path.home())
    if use_sonic:
        home_path = home_path + "/scratch"
    base_path = os.path.join(home_path, dl_config["BASE_PATH"])
    input_data_path = os.path.join(base_path, dl_config["INPUT_DATA_PATH"])

    exercise = dl_config["EXERCISE"]
    classifier_type = dl_config["CLASSIFIER_TYPE"]
    nb_epochs = dl_config.as_int("EPOCHS")
    multiclass_dir = dl_config["MULTICLASS_DIR"]
    output_path = os.path.join(home_path, dl_config["OUTPUT_PATH"])
    engine_prefix = dl_config["ENGINE_PREFIX"]
    sonic_prefix = dl_config["SONIC_PREFIX"]
    local_prefix = dl_config["LOCAL_PREFIX"]

    seed_value = dl_config["SEED_VALUES"]

    # if use_engine:
    #     input_data_path = input_data_path.replace(local_prefix, engine_prefix)
    #     output_path = output_path.replace(local_prefix, engine_prefix)
    # if use_sonic:
    #     input_data_path = input_data_path.replace(local_prefix, sonic_prefix)
    #     output_path = output_path.replace(local_prefix, sonic_prefix)

    logger.info("Input data path: {}, output data path: {}".format(input_data_path, output_path))
    logger.info("The seed value: {} ".format(seed_value))

    data_path = os.path.join(input_data_path, exercise, seed_value, multiclass_dir)
    weights_directory = os.path.join(output_path, 'dl_weights', exercise, classifier_type, multiclass_dir,
                                     seed_value)
    create_directory_if_not_exists(weights_directory)

    x_full_dataset, y_full_dataset, unique_pids, pid_full_df = combine_data(data_path)

    loocv_dl(x_full_dataset, y_full_dataset, unique_pids, pid_full_df, exercise)


"""
conda activate ml_utils
PROJECT_PATH=/home/ashish/Research/Codes/human_pose_estimation
export PYTHONPATH=$PYTHONPATH:$PROJECT_PATH
cd $PROJECT_PATH/time_series_classifiers/deep_learning_classifiers
python dl_cross_val.py --dl_config $PROJECT_PATH/time_series_classifiers/deep_learning_classifiers/dl_cross_val_config


"""
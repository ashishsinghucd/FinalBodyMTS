import argparse
import configparser
import os
import logging
import sys
from pathlib import Path
import time
import pickle

from configobj import ConfigObj
from sklearn import metrics
import numpy as np
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV
from sktime.utils.data_io import load_from_tsfile_to_dataframe
import sktime
import pandas as pd

from utils.program_stats import timeit
from utils.sklearn_utils import report_average, plot_confusion_matrix
from utils.util_functions import create_directory_if_not_exists

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FILE_NAME_X = '{}_{}_X'
FILE_NAME_Y = '{}_{}_Y'
FILE_NAME_PID = '{}_{}_pid'


def read_dataset(path):
    x_train, y_train = load_from_tsfile_to_dataframe(os.path.join(path,
                                                                  FILE_NAME_X.format("TRAIN", data_type) + ".ts"))

    logger.info("Training data shape {} {} {}".format(x_train.shape, len(x_train.iloc[0, 0]), y_train.shape))
    x_test, y_test = load_from_tsfile_to_dataframe(os.path.join(path,
                                                                FILE_NAME_X.format("TEST", data_type) + ".ts"))
    logger.info("Testing data shape: {} {}".format(x_test.shape, y_test.shape))

    test_pid = np.load(os.path.join(path, FILE_NAME_PID.format("TEST", data_type) + ".npy"), allow_pickle=True)
    train_pid = np.load(os.path.join(path, FILE_NAME_PID.format("TRAIN", data_type) + ".npy"), allow_pickle=True)

    x_full_dataset = pd.concat([x_train, x_test], axis=0)
    y_full_dataset = np.concatenate([y_train, y_test], axis=0)

    pid_full = np.concatenate([train_pid, test_pid], axis=0)
    logger.info("Full data shape: {} {} {}".format(x_full_dataset.shape, y_full_dataset.shape, pid_full.shape))
    unique_pids = np.unique([i.split("_")[0] for i in pid_full[:, 0]])
    pid_full_df = pd.DataFrame(pid_full, columns=["file_name", "start_time", "end_time", "rep_number"])
    pid_full_df["pid"] = pid_full_df["file_name"].apply(lambda x: x.split("_")[0])

    return x_full_dataset, y_full_dataset, unique_pids, pid_full_df


class RocketTransformerClassifier:
    def __init__(self, exercise):
        self.exercise = exercise
        self.classifiers_mapping = {}

    def fit_rocket(self, x_train, y_train, kernels=10000):
        rocket = Rocket(num_kernels=kernels, normalise=False)
        rocket.fit(x_train)
        x_training_transform = rocket.transform(x_train)
        self.classifiers_mapping["transformer"] = rocket
        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
        classifier.fit(x_training_transform, y_train)
        self.classifiers_mapping["classifier"] = classifier

    def predict_rocket(self, x_test, y_test):
        rocket = self.classifiers_mapping["transformer"]
        classifier = self.classifiers_mapping["classifier"]
        x_test_transform = rocket.transform(x_test)

        # Test Predictions
        predictions = classifier.predict(x_test_transform)
        accuracy = metrics.accuracy_score(y_test, predictions)
        return accuracy


def generate_cross_val_splits(pid):
    logger.info("Running for pid : {}".format(pid))
    test_ix = np.where(pid_full_df["pid"] == pid)[0]
    x_test = x_full_dataset.iloc[test_ix, :]
    y_test = y_full_dataset[test_ix]
    train_ix = np.argwhere(~pid_full_df.index.isin(test_ix)).reshape(-1)
    x_train = x_full_dataset.iloc[train_ix, :]
    y_train = y_full_dataset[train_ix]
    return x_train, y_train, x_test, y_test


def loocv_rocket():
    scores = []
    accuracy_scores_mapping = {"pid": [], "score": []}
    rocket_classifier = RocketTransformerClassifier(exercise)

    for pid in unique_pids:
        accuracy_scores_mapping["pid"].append(pid)
        x_train, y_train, x_test, y_test = generate_cross_val_splits(pid)
        logger.info("Training shape: {} {}".format(x_train.shape, y_train.shape))
        logger.info("Testing shape: {} {}".format(x_test.shape, y_test.shape))

        rocket_classifier.fit_rocket(x_train, y_train)
        accuracy = rocket_classifier.predict_rocket(x_test, y_test)
        scores.append(accuracy)
        accuracy_scores_mapping["score"].append(accuracy)
        logger.info("The accuracy: {} ".format(accuracy))
    accuracy_scores_mapping_df = pd.DataFrame(accuracy_scores_mapping)
    accuracy_scores_mapping_df.to_csv("/tmp/accuracy_cross_val_hpe.csv", index=False)
    logger.info("The average accuracy: {} std: {}".format(np.mean(scores), np.std(scores)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rocket_config", required=True, help="path of the config file")
    args = parser.parse_args()
    rocket_config = ConfigObj(args.rocket_config)

    home_path = str(Path.home())
    seed_values = rocket_config["SEED_VALUES"]
    exercise = rocket_config["EXERCISE"]
    output_path = os.path.join(home_path, rocket_config["OUTPUT_PATH"])
    data_type = rocket_config["DATA_TYPE"]

    base_path = os.path.join(home_path, rocket_config["BASE_PATH"], exercise)
    input_data_path = os.path.join(base_path, rocket_config["INPUT_DATA_PATH"])

    output_results_path = os.path.join(output_path, "RocketCrossVal")
    create_directory_if_not_exists(output_results_path)

    seed_value = 103007
    logger.info("----------------------------------------------------")
    logger.info("Performing Rocket Cross Validation for seed value: {}".format(seed_value))
    input_path_combined = os.path.join(input_data_path, str(seed_value), "MulticlassSplit")
    if not os.path.exists(input_path_combined):
        logger.info("Path does not exist for seed: {}".format(seed_value))
        sys.exit(1)
    x_full_dataset, y_full_dataset, unique_pids, pid_full_df = read_dataset(input_path_combined)
    ts = time.time()
    loocv_rocket()
    te = time.time()
    total_time = (te - ts)
    logger.info('Total time preprocessing: {} seconds'.format(total_time))


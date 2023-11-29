import argparse
import configparser
import os
import logging
from pathlib import Path

from configobj import ConfigObj
from sklearn import metrics
import numpy as np
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV
from sktime.utils.data_io import load_from_tsfile_to_dataframe
import sktime
import pandas as pd
from scipy.stats import mode
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV


from data_processing.extract_segments import TRAIN_DATASET_X, TEST_DATASET_X, VAL_DATASET_X, TRAIN_PID, TEST_PID
from utils.math_funtions import get_combinations
from utils.program_stats import timeit
from utils.sklearn_utils import report_average, plot_confusion_matrix
from utils.util_functions import create_directory_if_not_exists
from sktime.utils.data_processing import from_nested_to_3d_numpy, from_3d_numpy_to_nested

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FILE_NAME_X = '{}_{}_X'
FILE_NAME_Y = '{}_{}_Y'
FILE_NAME_PID = '{}_{}_pid'


def merge_prob_confidence(probs_df):
    # probs_df.set_index("PID", inplace=True)
    body_25_df = pd.read_csv(os.path.join(home_path, segment_stats_file))
    merge_df = pd.merge(probs_df, body_25_df, how='inner', on=["PID", "rep"])
    logger.info("Merge df shape: {}".format(merge_df.shape))
    return merge_df


def read_dataset(path):
    x_train, y_train = load_from_tsfile_to_dataframe(os.path.join(path,
                                                                  FILE_NAME_X.format("TRAIN", data_type) + ".ts"))

    """
    x_train_default, y_train_default = load_from_tsfile_to_dataframe(os.path.join(path,
                                                                  FILE_NAME_X.format("TRAIN", "default") + ".ts"))
    x_train = pd.concat([x_train, x_train_default], axis=0)
    y_train = np.concatenate([y_train_default, y_train_default])

    """
    logger.info("Training data shape {} {} {}".format(x_train.shape, len(x_train.iloc[0, 0]), y_train.shape))
    x_test, y_test = load_from_tsfile_to_dataframe(os.path.join(path,
                                                                FILE_NAME_X.format("TEST", data_type) + ".ts"))
    logger.info("Testing data shape: {} {}".format(x_test.shape, y_test.shape))


    # x_test, y_test = load_from_tsfile_to_dataframe(os.path.join(path,
    #                                                             FILE_NAME_X.format("TEST", "crf34") + ".ts"))
    logger.info("Testing data shape: {} {}".format(x_test.shape, y_test.shape))
    test_pid = np.load(os.path.join(path, FILE_NAME_PID.format("TEST", data_type) + ".npy"), allow_pickle=True)
    train_pid = np.load(os.path.join(path, FILE_NAME_PID.format("TRAIN", data_type) + ".npy"), allow_pickle=True)


    try:
        x_val, y_val = load_from_tsfile_to_dataframe(os.path.join(input_data_path,
                                                                  FILE_NAME_X.format("VAL", data_type) + ".ts"))
        logger.info("Validation data shape: {} {}".format(x_val.shape, y_val.shape))
    except (sktime.utils.data_io.TsFileParseException, FileNotFoundError):
        logger.info("Validation data is empty:")
        x_val, y_val = None, None

    return x_train, y_train, x_test, y_test, x_val, y_val, train_pid, test_pid


class EnsembleRocketTransformerClassifier:
    def __init__(self, exercise, ensize=20):
        self.exercise = exercise
        self.ensize = ensize
        self.starts = np.random.randint(20, size=self.ensize)
        self.ends = self.starts + 50 + np.random.randint(5, size=self.ensize)
        logger.info("Start and the ending indexes")
        logger.info(self.starts)
        logger.info(self.ends)
        self.classifiers_mapping = []

    @timeit
    def fit_rocket(self, x_train, y_train, train_pid, kernels=10000):
        x_train_3d = from_nested_to_3d_numpy(x_train)
        for start, end in zip(self.starts, self.ends):
            x_train_3d_sliced = x_train_3d[:, :, start:end]
            x_train_nested = from_3d_numpy_to_nested(x_train_3d_sliced)

            rocket = Rocket(num_kernels=kernels, normalise=False)    # random_state=100343
            rocket.fit(x_train_nested)
            x_training_transform = rocket.transform(x_train_nested)

            classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
            classifier.fit(x_training_transform, y_train)
            self.classifiers_mapping.append([rocket, classifier])

    @timeit
    def predict_rocket(self, x_test, y_test, test_pid, x_val=None, y_val=None):
        y_pred_list = []
        x_test_3d = from_nested_to_3d_numpy(x_test)
        l = len(np.unique(y_test))
        probs = np.zeros((x_test.shape[0], l))

        for start, end, cm in zip(self.starts, self.ends, self.classifiers_mapping):
            x_test_3d_sliced = x_test_3d[:, :, start:end]
            x_test_nested = from_3d_numpy_to_nested(x_test_3d_sliced)
            rocket = cm[0]
            classifier = cm[1]

            x_test_transform = rocket.transform(x_test_nested)
            predictions = classifier.predict(x_test_transform)
            y_pred_list.append(predictions)
            d = classifier.decision_function(x_test_transform)
            probs += np.exp(d) / np.sum(np.exp(d), axis=1).reshape(-1, 1)


        # Confusion Matrix

        final_predictions = mode(y_pred_list)[0][0]

        labels = list(np.sort(np.unique(y_test)))
        confusion_matrix = metrics.confusion_matrix(y_test, final_predictions)
        classification_report = metrics.classification_report(y_test, final_predictions)
        logger.info("-----------------------------------------------")
        logger.info("Metrics on testing data")
        logger.info("Accuracy {}".format(metrics.accuracy_score(y_test, final_predictions)))
        logger.info("\n Confusion Matrix: \n {}".format(confusion_matrix))
        logger.info("\n Classification report: \n{}".format(classification_report))

        classification_report_list.append(classification_report)

        plot_confusion_matrix(output_results_path, seed_value, confusion_matrix, labels)

        if x_val:
            logger.info("-----------------------------------------------")
            logger.info("Metrics on validation data")
            x_val_transform = rocket.transform(x_val)
            predictions = classifier.predict(x_val_transform)
            confusion_matrix = metrics.confusion_matrix(y_val, predictions)
            classification_report = metrics.classification_report(y_val, predictions)
            logger.info("Accuracy {}".format(metrics.accuracy_score(y_test, predictions)))
            logger.info("\n Confusion Matrix: \n {}".format(confusion_matrix))
            logger.info("\n Classification report: \n{}".format(classification_report))

    def create_prob_df(self, data_pid, y_test, predictions, probs, training_data=False):
        predictions_df = pd.DataFrame(probs, columns=self.classifiers_mapping["classifier"].classes_)
        predictions_df["PredictedLabel"] = predictions
        predictions_df["ActualLabel"] = y_test

        pid_info_df = pd.DataFrame(data_pid, columns=["Filename", "StartTime", "EndTime", "rep"])
        pid_info_df["PID"] = pid_info_df.apply(lambda x: x["Filename"].split(" ")[0], axis=1)
        pid_info_df["CorrectPrediction"] = y_test == predictions
        final_df = pd.concat([pid_info_df, predictions_df], axis=1)
        final_df = final_df.drop(["StartTime", "EndTime", "Filename"], axis=1)
        final_df["rep"] = final_df["rep"].astype(np.int64)
        f = "testing"
        if training_data:
            f = "training"
        merge_df = merge_prob_confidence(final_df)
        merge_df.to_csv('{}/probs_{}_{}.csv'.format(output_results_path, seed_value, f), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rocket_config", required=True, help="path of the config file")
    parser.add_argument("--exercise_config", required=True, help="path of the config file")
    args = parser.parse_args()
    rocket_config = ConfigObj(args.rocket_config)

    home_path = str(Path.home())
    base_path = os.path.join(home_path, rocket_config["BASE_PATH"])
    seed_values = rocket_config["SEED_VALUES"]
    input_data_path = os.path.join(base_path, rocket_config["INPUT_DATA_PATH"])
    exercise = rocket_config["EXERCISE"]
    multiclass_classification = rocket_config.as_bool("MULTICLASS_CLASSIFICATION")
    multiclass_dir = rocket_config["MULTICLASS_DIR"]
    output_path = os.path.join(home_path, rocket_config["OUTPUT_PATH"])
    normal_vs_abnormal = rocket_config.as_bool("NORMAL_VS_ABNORMAL")
    only_abnormal_classes = rocket_config.as_bool("ONLY_ABNORMAL_CLASSES")
    data_type = rocket_config["DATA_TYPE"]
    segment_stats_file = os.path.join(home_path, rocket_config["SEGMENT_STATS_FILE"]).format(data_type)
    config_parser = configparser.RawConfigParser()
    config_parser.read(args.exercise_config)
    valid_classes = config_parser.get(exercise, "valid_classes").split(",")
    label_index_mapping = {i + 1: value for i, value in enumerate(valid_classes)}
    index_label_mapping = {value: i + 1 for i, value in enumerate(valid_classes)}

    output_results_path = os.path.join(output_path, "Rocket")
    create_directory_if_not_exists(output_results_path)

    classification_report_list = []
    combination = multiclass_dir
    for seed_value in seed_values[1:2]:
        logger.info("----------------------------------------------------")
        logger.info("Fitting Ensemble Rocket for seed value: {}".format(seed_value))
        input_path_combined = os.path.join(input_data_path, exercise, seed_value, combination)
        if not os.path.exists(input_path_combined):
            logger.info("Path does not exist for seed: {}".format(seed_value))
            continue
        x_train, y_train, x_test, y_test, x_val, y_val, train_pid, test_pid = read_dataset(input_path_combined)
        classes_list = combination.split("vs")
        """
        kernels = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
        for kernel_size in [110000, 140000, 150000, 200000]:
            print("Kernel size : {}".format(kernel_size))
            rocket_classifier = RocketTransformerClassifier(exercise)
            rocket_classifier.fit_rocket(x_train, y_train, kernels=kernel_size)
            rocket_classifier.predict_rocket(x_test, y_test, x_val, y_val)
            print("----------------------------------------------------")
        """

        rocket_classifier = EnsembleRocketTransformerClassifier(exercise)
        rocket_classifier.fit_rocket(x_train, y_train, train_pid)
        rocket_classifier.predict_rocket(x_test, y_test, test_pid, x_val, y_val)

    logger.info("Average classification report")
    logger.info(report_average(*classification_report_list))



"""




"""
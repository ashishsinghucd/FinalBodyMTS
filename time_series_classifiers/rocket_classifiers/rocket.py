import argparse
import configparser
import os
import logging
from pathlib import Path
import time
import pickle
from collections import Counter

from configobj import ConfigObj
from sklearn import metrics
import numpy as np
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV
# from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.datasets import load_from_tsfile_to_dataframe
import sktime
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

from utils.program_stats import timeit
from utils.sklearn_utils import report_average, plot_confusion_matrix
from utils.util_functions import create_directory_if_not_exists

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Creating file name conventions
FILE_NAME_X = '{}_{}_X'
FILE_NAME_Y = '{}_{}_Y'
FILE_NAME_PID = '{}_{}_pid'
exercise_types_mapping = {"MP": ["A", "Arch", "N", "R"], "Rowing": ["A", "Ext", "N", "R", "RB"]}


def merge_prob_confidence(probs_df):
    """
    Deprecated function
    :param probs_df:
    :return:
    """
    # probs_df.set_index("PID", inplace=True)
    body_25_df = pd.read_csv(os.path.join(home_path, segment_stats_file))
    merge_df = pd.merge(probs_df, body_25_df, how='inner', on=["PID", "rep"])
    logger.info("Merge df shape: {}".format(merge_df.shape))
    return merge_df


def read_dataset(path):
    """
    This function reads the data in sktime format for train/test/val
    :param path: path to the directory
    :return:
    """
    x_train, y_train = load_from_tsfile_to_dataframe(os.path.join(path,
                                                                  FILE_NAME_X.format("TRAIN", data_type) + ".ts"))

    logger.info("Training data shape {} {} {}".format(x_train.shape, len(x_train.iloc[0, 0]), y_train.shape))
    x_test, y_test = load_from_tsfile_to_dataframe(os.path.join(path,
                                                                FILE_NAME_X.format("TEST", data_type) + ".ts"))
    logger.info("Testing data shape: {} {}".format(x_test.shape, y_test.shape))

    logger.info("Testing data shape: {} {}".format(x_test.shape, y_test.shape))
    test_pid = np.load(os.path.join(path, FILE_NAME_PID.format("TEST", data_type) + ".npy"), allow_pickle=True)
    train_pid = np.load(os.path.join(path, FILE_NAME_PID.format("TRAIN", data_type) + ".npy"), allow_pickle=True)

    try:
        x_val, y_val = load_from_tsfile_to_dataframe(os.path.join(input_data_path,
                                                                  FILE_NAME_X.format("VAL", data_type) + ".ts"))
        logger.info("Validation data shape: {} {}".format(x_val.shape, y_val.shape))
    except FileNotFoundError:
        logger.info("Validation data is empty:")
        x_val, y_val = None, None

    return x_train, y_train, x_test, y_test, x_val, y_val, train_pid, test_pid


def make_binary_data(x_train, y_train, x_test, y_test):
    y_train_new = np.array(["normal" if i == "n" else "wrong" for i in y_train])
    y_test_new = np.array(["normal" if i == "n" else "wrong" for i in y_test])

    undersample = RandomUnderSampler(sampling_strategy='majority')
    x_over, y_over = undersample.fit_resample(x_train, y_train_new)

    logger.info("After undersampling training data shape: {} {}".format(x_over.shape, y_over.shape))

    return x_over, y_over, x_test, y_test_new


class RocketTransformerClassifier:
    def __init__(self, exercise):
        """
        Init function to initialize the exercise and dict for saving the models
        :param exercise:
        """
        self.exercise = exercise
        self.classifiers_mapping = {}

    @timeit
    def fit_rocket(self, x_train, y_train, train_pid, kernels=10000):
        """
        Fit the rocket transformer along with the Ridge Classifier.
        :param x_train:
        :param y_train:
        :param train_pid: pid file containing the pid information for each sample.
        :param kernels:
        :return:
        """
        rocket = Rocket(num_kernels=kernels, normalise=False)  # random_state=100343
        rocket.fit(x_train)
        x_training_transform = rocket.transform(x_train)
        self.classifiers_mapping["transformer"] = rocket
        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)

        # train_pid_df stores the filename, start time, end time and repetition number for a particular execution type
        train_pid_df = pd.DataFrame(train_pid, columns=["Filename", "StartTime", "EndTime", "rep"])
        train_pid_df["PID"] = train_pid_df.apply(lambda x: x["Filename"].split("_")[0], axis=1)
        train_demo_features_df = pd.merge(train_pid_df, gender_info_df, left_on="PID", right_on="Participant")
        train_demo_features_df = train_demo_features_df.loc[:, ["Sex", "Age", "Height", "Weight", "BMI"]]
        train_demo_features_df["Sex"] = train_demo_features_df["Sex"].apply(lambda x: 1 if x == "F" else 0)
        logger.info("Demo features combined data shape: {} {}".format(train_demo_features_df.shape, y_train.shape))

        # The above data can be combined with the transformed feature from rocket to create additional features
        # x_training_transform_combined = pd.concat([x_training_transform, train_demo_features_df], axis=1)
        # logger.info("Train and demo features combined data shape: {} {}".format(x_training_transform_combined.shape, y_train.shape))
        classifier.fit(x_training_transform, y_train)

        # Training Predictions
        # predictions = classifier.predict(x_training_transform)
        # d = classifier.decision_function(x_training_transform)
        # probs = np.exp(d) / np.sum(np.exp(d), axis=1).reshape(-1, 1)

        # Use this dict to save the model for testing later on
        self.classifiers_mapping["classifier"] = classifier

        if save_model:
            with open('/tmp/rocket_transformer.pkl', 'wb') as f:
                pickle.dump(rocket, f)

            with open('/tmp/ridge_model.pkl', 'wb') as f:
                pickle.dump(classifier, f)
        # Use this function to map the probabilities values for further analysis
        # self.create_prob_df(train_pid, y_train, predictions, probs, training_data=True)

    @timeit
    def predict_rocket(self, x_test, y_test, test_pid, x_val=None, y_val=None):
        """
        Functon to classify the test data using the ROCKET
        :param x_test:
        :param y_test:
        :param test_pid: file containing info of the test pids such as start and end time or repetition number
        :param x_val:
        :param y_val:
        :return:
        """
        rocket = self.classifiers_mapping["transformer"]
        classifier = self.classifiers_mapping["classifier"]
        x_test_transform = rocket.transform(x_test)

        # test_pid_df stores the filename, start time, end time and repetition number for a particular execution type
        test_pid_df = pd.DataFrame(test_pid, columns=["Filename", "StartTime", "EndTime", "rep"])
        test_pid_df["PID"] = test_pid_df.apply(lambda x: x["Filename"].split("_")[0], axis=1)
        test_demo_features_df = pd.merge(test_pid_df, gender_info_df, left_on="PID", right_on="Participant")
        test_demo_features_df = test_demo_features_df.loc[:, ["Sex", "Age", "Height", "Weight", "BMI"]]
        test_demo_features_df["Sex"] = test_demo_features_df["Sex"].apply(lambda x: 1 if x == "F" else 0)
        logger.info("Demo features combined data shape: {} {}".format(test_demo_features_df.shape, y_test.shape))
        #
        # x_test_transform_combined = pd.concat([x_test_transform, test_demo_features_df], axis=1)
        # logger.info("Test and demo features combined data shape: {} {}".format(x_test_transform_combined.shape,
        #                                                                         y_test.shape))

        # Test Predictions
        predictions = classifier.predict(x_test_transform)
        # d = self.classifiers_mapping["classifier"].decision_function(x_test_transform)
        # probs = np.exp(d) / np.sum(np.exp(d), axis=1).reshape(-1, 1)

        # self.calc_error_rate(test_pid, y_test, predictions)

        # Confusion Matrix
        labels = list(np.sort(np.unique(y_test)))
        confusion_matrix = metrics.confusion_matrix(y_test, predictions)
        classification_report = metrics.classification_report(y_test, predictions)
        logger.info("-----------------------------------------------")
        logger.info("Metrics on testing data")
        logger.info("Accuracy {}".format(metrics.accuracy_score(y_test, predictions)))
        logger.info("\n Confusion Matrix: \n {}".format(confusion_matrix))
        logger.info("\n Classification report: \n{}".format(classification_report))

        classification_report_list.append(classification_report)
        plot_confusion_matrix(output_results_path, seed_value, confusion_matrix, labels)

        # If we have the validation data
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

    def calc_error_rate(self, data_pid, y_test, predictions, file_path="/tmp/pid_tempinfo.csv"):
        """
        This function is used to calculate the additional info on the test data such as generating confusing matrix
        gender wise, or age wise.
        :param file_path:
        :param data_pid:
        :param y_test:
        :param predictions:
        :return:
        """
        pid_info_df = pd.DataFrame(data_pid, columns=["Filename", "StartTime", "EndTime", "rep"])
        pid_info_df["PID"] = pid_info_df.apply(lambda x: x["Filename"].split("_")[0], axis=1)
        pid_info_df["CorrectPrediction"] = y_test == predictions

        pid_info_df = pd.merge(pid_info_df, gender_info_df, left_on="PID", right_on="Participant")
        pid_info_df.to_csv(file_path, index=False)
        unique_participants = np.unique(pid_info_df["PID"])
        participants_gender = []

        gender_info = dict(zip(gender_info_df.Participant, gender_info_df.Sex))

        for i in list(unique_participants):
            participants_gender.append(gender_info[i])

        gender_info_list = []
        for i in list(pid_info_df["PID"]):
            gender_info_list.append(gender_info[i])

        logger.info("Test data gender distribution are: {}".format(str(Counter(participants_gender))))
        pid_info_df["Gender"] = gender_info_list

        male_participants = pid_info_df["Gender"] == "M"
        female_participants = pid_info_df["Gender"] == "F"

        confusion_matrix_males = metrics.confusion_matrix(y_test[male_participants], predictions[male_participants])
        confusion_matrix_females = metrics.confusion_matrix(y_test[female_participants],
                                                            predictions[female_participants])

        logger.info("\n Confusion Matrix Females: \n {}".format(confusion_matrix_females))
        logger.info("\n Confusion Matrix Males: \n {}".format(confusion_matrix_males))
        logging.info(pid_info_df.groupby(["Gender", "CorrectPrediction"]).size().reset_index())

    def create_prob_df(self, data_pid, y_test, predictions, probs, training_data=False):
        """
        Function to generate additional info such as probabilities for each sample
        :param data_pid:
        :param y_test:
        :param predictions:
        :param probs:
        :param training_data:
        :return:
        """
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
        final_df.to_csv('{}/probs_{}_{}.csv'.format(output_results_path, seed_value, f), index=False)
        # merge_df = merge_prob_confidence(final_df)
        # merge_df.to_csv('{}/probs_{}_{}.csv'.format(output_results_path, seed_value, f), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rocket_config", required=True, help="path of the config file")
    parser.add_argument("--exercise_config", required=True, help="path of the config file")
    args = parser.parse_args()
    rocket_config = ConfigObj(args.rocket_config)

    home_path = str(Path.home())
    seed_values = rocket_config["SEED_VALUES"]
    exercise = rocket_config["EXERCISE"]
    output_path = os.path.join(home_path, rocket_config["OUTPUT_PATH"])
    data_type = rocket_config["DATA_TYPE"]
    binary_classification = rocket_config.as_bool("BINARY_CLASSIFICATION")
    save_model = rocket_config.as_bool("SAVE_MODEL")
    gender_info_file = rocket_config["GENDER_INFO"]
    config_parser = configparser.RawConfigParser()
    config_parser.read(args.exercise_config)
    valid_classes = config_parser.get(exercise, "valid_classes").split(",")
    label_index_mapping = {i + 1: value for i, value in enumerate(valid_classes)}
    index_label_mapping = {value: i + 1 for i, value in enumerate(valid_classes)}

    base_path = os.path.join(home_path, rocket_config["BASE_PATH"], exercise)
    input_data_path = os.path.join(base_path, rocket_config["INPUT_DATA_PATH"])
    # segment_stats_file = os.path.join(home_path, rocket_config["SEGMENT_STATS_FILE"]).format(data_type)

    output_results_path = os.path.join(output_path, "Rocket")
    create_directory_if_not_exists(output_results_path)

    gender_info_df = pd.read_csv(gender_info_file, sep=",")

    classification_report_list = []
    for seed_value in seed_values:
        logger.info("----------------------------------------------------")
        logger.info("Fitting Rocket for seed value: {}".format(seed_value))
        input_path_combined = os.path.join(input_data_path, seed_value, "MulticlassSplit")
        if not os.path.exists(input_path_combined):
            logger.info("Path does not exist for seed: {}".format(seed_value))
            continue
        x_train, y_train, x_test, y_test, x_val, y_val, train_pid, test_pid = read_dataset(input_path_combined)
        if binary_classification:
            x_train, y_train, x_test, y_test = make_binary_data(x_train, y_train, x_test, y_test)
        ts = time.time()
        rocket_classifier = RocketTransformerClassifier(exercise)
        rocket_classifier.fit_rocket(x_train, y_train, train_pid)
        rocket_classifier.predict_rocket(x_test, y_test, test_pid, x_val, y_val)
        te = time.time()
        total_time = (te - ts)
        logger.info('Total time preprocessing: {} seconds'.format(total_time))

    logger.info("Average classification report")
    # Print average classification report over 3 splits
    logger.info(report_average(*classification_report_list))

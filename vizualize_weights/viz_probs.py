import os
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow.keras as keras
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns

from vizualize_weights.read_data_utils import read_datasets_numpy

sns.set_style("dark")
from sklearn import metrics
from configobj import ConfigObj

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_probs(data_x, data_y, data_pid):
    predictions = model.predict(data_x)
    y_pred = np.argmax(predictions, axis=1)
    y_pred_label = np.array([enc.categories_[0][i] for i in y_pred])
    predictions_df = pd.DataFrame(predictions, columns=enc.categories_[0])
    predictions_df["PredictedLabel"] = y_pred_label
    predictions_df["ActualLabel"] = data_y
    # prob_df = pd.concat()

    pid_info_df = pd.DataFrame(data_pid, columns=["Filename", "StartTime", "EndTime"])
    pid_info_df["PID"] = pid_info_df.apply(lambda x: x["Filename"].split(" ")[0], axis=1)
    pid_info_df["CorrectPrediction"] = data_y == y_pred_label
    final_df = pd.concat([pid_info_df, predictions_df], axis=1)
    final_df.to_csv("/tmp/probs_classes_{}.csv".format(model_name), index=False)
    logger.info("Accuracy of the model is: {}".format(metrics.accuracy_score(data_y, y_pred_label)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path of the config file")
    args = parser.parse_args()
    viz_config = ConfigObj(args.config)

    home_path = str(Path.home())
    base_path = os.path.join(home_path, viz_config["BASE_PATH"])
    input_data_path = os.path.join(base_path, viz_config["INPUT_DATA_PATH"])

    exercise = viz_config["EXERCISE"]
    seed_value = viz_config["SEED_VALUE"]
    combination = viz_config["COMBINATION"]
    model_name = viz_config["MODEL_NAME"]
    model_path = os.path.join(home_path, viz_config["MODEL_PATH"].format(exercise, model_name, seed_value))
    pid_list = viz_config.as_list("PID_LIST")
    extracted_frames_path = os.path.join(base_path, viz_config["EXTRACTED_FRAMES_PATH"].format(exercise))

    input_path_combined = os.path.join(input_data_path, exercise, seed_value, combination)
    if not os.path.exists(input_path_combined):
        logger.info("Path does not exist for seed: {}".format(seed_value))

    x_train, y_train, train_pid, x_test, y_test, test_pid = read_datasets_numpy(input_path_combined)

    logger.info("Training data shape {} {}".format(x_train.shape, y_train.shape))
    logger.info("Testing data shape: {} {}".format(x_test.shape, y_test.shape))

    enc = OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))

    model = keras.models.load_model(model_path)

    logger.info(model.summary())

    length_ts = x_train.shape[1]
    dims = x_train.shape[2]
    logger.info("Length of single time series: {}, total dimensions: {}".format(length_ts, dims))

    generate_probs(x_test, y_test, test_pid)


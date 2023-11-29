import os
import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from sklearn.preprocessing import OneHotEncoder
from scipy.interpolate import interp1d
import seaborn as sns

from vizualize_weights.analyze_model import AnalyzeModel
from vizualize_weights.display_grid_frames import get_frames_path, plot_frames
from vizualize_weights.read_data_utils import read_datasets_numpy, get_custom_indices
from utils.math_funtions import get_top_values
from utils.util_functions import create_directory_if_not_exists, delete_directory_if_exists

sns.set_style("dark")
from configobj import ConfigObj

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRAIN_DATASET_X = "TRAIN_X"
TEST_DATASET_X = "TEST_X"
TRAIN_DATASET_Y = "TRAIN_Y"
TEST_DATASET_Y = "TEST_Y"
VAL_DATASET_Y = "VAL_Y"
VAL_DATASET_X = "VAL_X"
TRAIN_PID = "TRAIN_pid"
TEST_PID = "TEST_pid"
BODY_PARTS_ORDER = ['RShoulder_X', 'RElbow_X', 'RWrist_X', 'LShoulder_X', 'LElbow_X', 'LWrist_X', 'RHip_X', 'LHip_X',
                    'RShoulder_Y', 'RElbow_Y', 'RWrist_Y', 'LShoulder_Y', 'LElbow_Y', 'LWrist_Y', 'RHip_Y', 'LHip_Y']


class VizCam:
    """
    Class to generate and display the saliency map using CAM
    """
    def __init__(self, model, model_name, enc, length_ts, dims):
        """
        The output path will store the final important frames
        :param model:
        :param model_name: model name
        :param enc: encoder
        :param length_ts:length of the time series
        :param dims: number of dimensions
        """
        self.model = model
        self.model_name = model_name
        self.enc = enc
        self.last_layer_weights = self.model.layers[-1].get_weights()[0]
        self.cam_model = keras.Model(inputs=self.model.input,
                                     outputs=(self.model.layers[-3].output, self.model.layers[-1].output))
        self.length_ts = length_ts
        self.dims = dims

        self.output_path = "/tmp/{}".format(self.model_name)
        delete_directory_if_exists(self.output_path)
        create_directory_if_not_exists(self.output_path)

    def get_cam_single_sample(self, index_of_sample, data_x, data_y):
        """
        Generate CAM for a single repetition
        :param index_of_sample:
        :param data_x: data corresponding to a single repetition
        :param data_y:
        :return:
        """
        single_example, actual_label = data_x[index_of_sample], data_y[index_of_sample]
        single_example = np.expand_dims(single_example, axis=0)

        predicted_prob = self.model.predict(single_example)
        predicted_int = np.argmax(predicted_prob)
        predicted_label = self.enc.categories_[0][predicted_int]

        features_for_one_img, results = self.cam_model.predict(single_example)
        cam = np.zeros(dtype=np.float, shape=(features_for_one_img.shape[1]))
        for k, w in enumerate(self.last_layer_weights[:, predicted_int]):
            cam += w * features_for_one_img[0, :, k]
        logger.debug("Original CAM: {}".format(cam))

        logger.debug("CAM after normalization: {}".format(cam))
        return single_example, actual_label, predicted_label, predicted_prob, predicted_int, cam

    def resample_same_length(self, numpy_mat, body_part_ind, cam, length, plot_original_length):
        """
        Function to resample time to a fixed length
        :param numpy_mat:
        :param body_part_ind:
        :param cam:
        :param length:
        :param plot_original_length:
        :return:
        """
        x = np.linspace(0, self.length_ts - 1, length, endpoint=True)
        f = interp1d(range(self.length_ts), numpy_mat[0, :, body_part_ind].squeeze())
        y = f(x)
        f = interp1d(range(self.length_ts), cam)
        cam1 = f(x).astype(int)
        if plot_original_length:
            return y, cam1, length
        return numpy_mat[0, :, body_part_ind].squeeze(), cam, self.length_ts

    @staticmethod
    def normalize_cam(cam):
        """
        Function to normalize the CAM values
        :param cam:
        :return:
        """
        minimum = np.min(cam)
        cam = cam - minimum
        cam = cam / max(cam)
        cam = cam * 100
        return cam

    def display_single_example_cam(self, index_of_sample, data_x, data_y, data_pid,
                                   plot_original_length=True, figsize=(20, 15), cmap='jet'):
        """
        Function to display CAM values in a graph
        :param index_of_sample:
        :param data_x:
        :param data_y:
        :param data_pid:
        :param plot_original_length:
        :param figsize:
        :param cmap:
        :return:
        """
        logger.info("The sample info: {} {} {}".format(data_pid[index_of_sample][0], data_pid[index_of_sample][1],
                                                       data_pid[index_of_sample][2]))

        start_time, end_time = float(data_pid[index_of_sample][1]), float(data_pid[index_of_sample][2])
        length = int(end_time - start_time)

        single_example, actual_label, predicted_label, predicted_prob, predicted_int, original_cam = \
            self.get_cam_single_sample(index_of_sample, data_x, data_y)
        numpy_mat = single_example

        frame_indices_not_cal = False
        fig = plt.figure(figsize=figsize)
        for i, body_part_ind in enumerate([0, 8, 1, 9, 2, 10, 6, 14]):  # range(0, self.dims, 2)
            cam = original_cam
            ax = fig.add_subplot(4, 2, i + 1)
            y, cam, final_length = self.resample_same_length(numpy_mat, body_part_ind, cam, length,
                                                             plot_original_length)

            cam = self.normalize_cam(cam)

            if not frame_indices_not_cal:
                top_idx, top_values = get_top_values(cam, top_k)
                frame_indices_not_cal = True
            y = (y - y.mean()) / y.std()
            ax.plot(y)
            t = ax.scatter(np.arange(final_length), y=y, cmap=cmap, c=cam, s=100)  # jet or hot_r
            ax.set_title("Body Part: {}".format(BODY_PARTS_ORDER[body_part_ind]))
            plt.xlabel("Index")
            plt.ylabel("Y-coordinates")
            plt.colorbar(t, ax=ax)

        plt.tight_layout()
        st = plt.suptitle(
            'True label:' + actual_label + ', Predicted label:' + predicted_label + ',  Likelihood of ' + predicted_label + ': ' + str(
                predicted_prob[0][predicted_int]))
        st.set_y(0.95)
        fig.subplots_adjust(top=0.85)
        file_name = data_pid[index_of_sample][0] + "_" + str(data_pid[index_of_sample][1]) + "_" + str(
            data_pid[index_of_sample][2])
        if actual_label != predicted_label:
            file_name += "_w"
        plt.savefig("{}/{}.jpg".format(self.output_path, file_name))
        plt.close(fig)
        logger.info("Top values in the CAM. Indices: {} Values: {}".format(top_idx, top_values))
        return actual_label, predicted_label, predicted_prob[0][predicted_int], top_idx, top_values

    def interpret_class_weights(self, data_x, data_y, body_part_ind=9, prob_threshold=0.9, samples_plot=80, cmap='jet',
                                figsize=(10, 5)):
        """
        Function to generate CAM for training data. It generates and displays CAM for fixed number of samples
        which are correctly classified
        :param data_x:
        :param data_y:
        :param body_part_ind: which body part to plot
        :param prob_threshold: probability threshold (only plot samples having correct prob greater than this value)
        :param samples_plot: number of correctly classified samples
        :param cmap:
        :param figsize:
        :return:
        """
        classes = np.unique(data_y)
        for c in classes:
            correct_predictions = 0
            fig = plt.figure(figsize=figsize)
            c_x_train = data_x[np.where(data_y == c)[0]]
            c_y_train = data_y[np.where(data_y == c)[0]]
            total_samples = c_x_train.shape[0]
            logger.info("Total samples of class {} are: {}".format(c, total_samples))
            logger.info("Plotting only {} samples out of total {}".format(samples_plot, total_samples))
            for ind in range(total_samples):

                single_example, actual_label, predicted_label, predicted_prob, predicted_int, cam = \
                    self.get_cam_single_sample(ind, c_x_train, c_y_train)

                predicted_prob = self.model.predict(single_example)
                prob = np.max(predicted_prob)
                if prob < prob_threshold:
                    continue
                self.argmax = np.argmax(predicted_prob)
                predicted_int = self.argmax
                predicted_label = self.enc.categories_[0][predicted_int]

                if predicted_label != c:
                    continue
                correct_predictions += 1

                if correct_predictions == samples_plot:
                    break

                t = plt.scatter(np.arange(length_ts), single_example[0, :, body_part_ind].squeeze(), cmap=cmap,
                                c=cam, marker='.', s=30)  # jet or hot_r
                # plt.title("Class : {} Total samples: {} Correctly predicted: {} Body Part: "
                #           "{}".format(c, total_samples, correct_predictions,
                #                       BODY_PARTS_ORDER[body_part_ind]))
                plt.title("Class : {} Body Part: "
                          "{}".format(c,
                                      BODY_PARTS_ORDER[body_part_ind]))
            plt.colorbar(t)

            plt.savefig("{}/class_weights_{}.jpg".format(self.output_path, c))
            plt.close(fig)


def generate_cam(data_x, data_y, data_pid):
    """
    Main function to generate, display and save saliency map using CAM
    :param data_x:
    :param data_y:
    :param data_pid:
    :return:
    """
    analyze_model = AnalyzeModel(model, data_x, data_y, data_pid, enc)
    class_instances_ind = analyze_model.get_indices_data(data_y)
    viz_cam_model = VizCam(model, model_name, enc, length_ts, dims)

    # List of pids to generate CAM for
    for pid in pid_list:
        count_mapping = {}
        logger.info("Running for the pid: {}".format(pid))
        indices = get_custom_indices(pid, analyze_model.pid_info)
        logger.info("Running for the indices: {}".format(indices))
        for index_sample in indices:
            clip_name = data_pid[index_sample][0]
            file_name = data_pid[index_sample][0] + "_" + str(data_pid[index_sample][1]) + "_" + str(
                data_pid[index_sample][2])
            if not clip_name.endswith("_R"):
                continue
            exercise_type = clip_name.split("_")[-2]
            if exercise_type not in count_mapping:
                count_mapping[exercise_type] = 0
            # if count_mapping[exercise_type] >= 1:
            #     continue
            logger.info("Running for the index: {}".format(index_sample))
            # This one only display the CAM values
            actual_label, predicted_label, predicted_prob, top_idx, top_values = viz_cam_model.display_single_example_cam(
                index_sample, data_x, data_y, data_pid, True)
            logger.info("High intensity regions: {}".format(top_idx))
            count_mapping[exercise_type] += 1

            # This one displays the actual frames
            frames_path, start_frame, end_frame = get_frames_path(index_sample, data_pid, extracted_frames_path)
            plot_frames(top_idx, top_values, frames_path, actual_label, predicted_label, predicted_prob, file_name,
                        start_frame, end_frame, viz_cam_model.output_path)

    # For training data visualization
    if interpret_class_weights:
        viz_cam_model.interpret_class_weights(data_x, data_y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--viz_config", required=True, help="path of the config file")
    args = parser.parse_args()
    viz_config = ConfigObj(args.viz_config)

    home_path = str(Path.home())
    base_path = os.path.join(home_path, viz_config["BASE_PATH"])
    exercise = viz_config["EXERCISE"]
    input_data_path = os.path.join(base_path, exercise, viz_config["INPUT_DATA_PATH"])

    data_type = viz_config["DATA_TYPE"]
    seed_value = viz_config["SEED_VALUE"]
    combination = viz_config["COMBINATION"]
    model_name = viz_config["MODEL_NAME"]
    model_path = os.path.join(home_path, viz_config["MODEL_PATH"].format(exercise, model_name, seed_value))
    index_sample = viz_config.as_int("INDEX_SAMPLE")
    count_display = viz_config.as_int("COUNT_DISPLAY")
    pid_list = viz_config.as_list("PID_LIST")
    top_k = viz_config.as_int("TOP_K")
    interpret_class_weights = viz_config.as_bool("INTERPRET_CLASS_WEIGHTS")
    extracted_frames_path = os.path.join(base_path, viz_config["EXTRACTED_FRAMES_PATH"].format(exercise))

    input_path_combined = os.path.join(input_data_path, seed_value, combination)
    if not os.path.exists(input_path_combined):
        logger.info("Path does not exist for seed: {}".format(seed_value))

    x_train, y_train, train_pid, x_test, y_test, test_pid = read_datasets_numpy(input_path_combined, data_type)

    logger.info("Training data shape {} {}".format(x_train.shape, y_train.shape))
    logger.info("Testing data shape: {} {}".format(x_test.shape, y_test.shape))

    enc = OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))

    model = keras.models.load_model(model_path)

    logger.info(model.summary())

    length_ts = x_train.shape[1]
    dims = x_train.shape[2]
    logger.info("Length of single time series: {}, total dimensions: {}".format(length_ts, dims))

    generate_cam(x_test, y_test, test_pid)
    # generate_cam(x_train, y_train, train_pid)

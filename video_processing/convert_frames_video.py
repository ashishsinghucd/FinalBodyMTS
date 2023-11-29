import argparse
import configparser
import getpass
import json
import os
import logging.config
import warnings
from pathlib import Path

import cv2
from configobj import ConfigObj
import numpy as np

from utils.common import get_exercise_config
from utils.util_functions import create_directory_if_not_exists

logging.config.fileConfig(fname='../configs/logging_config', disable_existing_loggers=False)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def extract_pid_type(splitted_path):
    pid, exercise_type, file_name = splitted_path[-3], splitted_path[-2], splitted_path[-1]
    frame_number = file_name.split(".")[0].split("_")[1]
    return exercise_type, file_name, frame_number, pid


def convert_frames_to_videos():
    frame_path_template = "{frame_path}/{pid}/{exercise_type}/{rep}/frame_{frame_number}.jpg"
    count = 0
    logger.info("Total pid's are: {}".format(len(frames_info)))
    for pid, frame_peak in frames_info.items():
        for exercise_type, peaks in frame_peak.items():
            if not os.path.exists(os.path.join(full_extracted_frames_path, pid, exercise_type)):
                continue
            logger.info("{} {}".format(pid, exercise_type))
            logger.info("Length of peaks {}".format(len(peaks)))
            try:
                for ind in range(len(peaks) - 1):
                    starting_frame = peaks[ind]
                    ending_frame = peaks[ind + 1]
                    codec = cv2.VideoWriter_fourcc(*'mp4v')  # ffmpeg -i P4_R_111_167.mp4  -vcodec libx264 output.mp4
                    output_file_name = "{}_{}_{}.mp4".format(pid, exercise_type, ind + 1)
                    output_video_path = os.path.join(full_converted_videos_path, output_file_name)
                    if os.path.exists(output_video_path):
                        continue
                    first_frame = cv2.imread(frame_path_template.format(frame_path=full_extracted_frames_path, pid=pid,
                                                                        exercise_type=exercise_type,
                                                                        rep=ind + 1,
                                                                        frame_number=0))
                    height, width, channels = first_frame.shape
                    out = cv2.VideoWriter(output_video_path, codec, frame_rate, (width, height))
                    for frame in range(starting_frame, ending_frame):
                        image = cv2.imread(frame_path_template.format(frame_path=full_extracted_frames_path, pid=pid,
                                                                      exercise_type=exercise_type,
                                                                      rep=ind + 1,
                                                                      frame_number=frame - starting_frame))
                        out.write(image)
                    out.release()
                    cv2.destroyAllWindows()
                    count += 1
                    if not count % 50:
                        logger.info("Written {} videos to disk".format(str(count)))
            except Exception as e:
                logger.exception("Error converting the frame into video: {}".format(str(e)))


def decide_machine():
    global use_sonic, base_path
    if getpass.getuser() != "ashish":
        use_sonic = True
    sonic_prefix = config["SONIC_PREFIX"]
    local_prefix = config["LOCAL_PREFIX"]
    if use_sonic:
        base_path = base_path.replace(local_prefix, sonic_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_config", required=True, help="path of the config file")
    parser.add_argument("--exercise_config", required=True, help="path of the config file")
    parser.add_argument("--peaks_mapping", required=True, help="path to the peaks mapping file")
    args = parser.parse_args()

    config = ConfigObj(args.video_config)

    home_path = str(Path.home())

    # Read the arguments
    exercise = config["EXERCISE"]
    # Read the paths and folders
    base_path = os.path.join(home_path, config["BASE_PATH"])
    extracted_frames_folder = config["EXTRACTED_FRAMES_FOLDER"]
    converted_videos_folder = config["CONVERTED_VIDEOS_FOLDER"]
    frame_rate = config.as_int("FRAME_RATE")
    use_sonic = False
    decide_machine()
    full_extracted_frames_path = os.path.join(base_path, extracted_frames_folder, exercise)
    full_converted_videos_path = os.path.join(base_path, converted_videos_folder, exercise)
    create_directory_if_not_exists(full_converted_videos_path)

    with open(args.peaks_mapping, "r") as f:
        frames_info = json.load(f)
    important_parts, peak_parts, valid_classes = get_exercise_config(exercise, args.exercise_config)

    convert_frames_to_videos()

"""

ap = argparse.ArgumentParser()
ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
args = vars(ap.parse_args())


Runtime: 6 hours for poor quality on crf 40

Runtime: 7:9:48 (hh:mm:ss)


"""

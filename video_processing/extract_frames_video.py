import os
import configparser
import logging.config
from pathlib import Path

import cv2

from utils.util_functions import create_directory_if_not_exists

logger = logging.getLogger(__name__)

def save_frame(save_frame_path, image, count):
    try:
        save_full_path = os.path.join(save_frame_path, "frame_{}.jpg".format(str(count)))
        if not os.path.exists(save_full_path):
            cv2.imwrite(save_full_path, image)
    except Exception as e:
        logger.exception("Error saving the frames: {}".format(str(e)))

def get_and_save_frame(clip_file_path, save_frame_path, start=-1, end=-1, every=1):
    file_name = clip_file_path.rsplit("/", 1)[1].rsplit(".", 1)[0]
    cam = cv2.VideoCapture(clip_file_path)
    if start < 0:
        start = 0
    if end < 0:
        end = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Total frames for {file_name} are: {end}")
    cam.set(1, start)
    count = 0
    frame = start
    while_safety = 0
    start_frame = 0
    end_frame = -1

    while frame < end:
        ret_val, input_image = cam.read()
        if while_safety > 500:
            break
        if input_image is None:
            while_safety += 1
            continue
        if count and count % 100 == 0:
            logger.info(f"Processed {count} frames")
        if count % every == 0:
            save_frame(save_frame_path, input_image, count)
            count += 1
        frame += 1
    cam.release()


home_path = str(Path.home())
exercise = "MP"
base_path = os.path.join(home_path, "Results/Datasets/HPE2/")

full_extracted_clips_path = os.path.join(base_path, "CRF46/MP")
full_extracted_frames_path = os.path.join(base_path, "FramesCRF46/MP")
create_directory_if_not_exists(full_extracted_frames_path)

count = 0
for root, dirs, files in os.walk(full_extracted_clips_path):
    for name in files:
        try:
            if name.startswith("."):
                continue
            logger.info("Extracting the frames of: {}".format(name))
            clip_file_path = os.path.join(root, name)
            split_info = name[:-4].split("_")
            pid, exercise_type, rep = split_info[0], split_info[1], split_info[2]
            folder_name = name[:-4]
            save_frame_path = os.path.join(full_extracted_frames_path, folder_name)
            create_directory_if_not_exists(save_frame_path)
            get_and_save_frame(clip_file_path, save_frame_path)
            count += 1
            if not count % 50:
                logger.info("Finished extracting frames for {} files".format(str(count)))
            # break
        except Exception as e:
            logger.exception("Error extracting the frames of: {} {}".format(name, str(e)))
            logger.exception("Exception occurred ")

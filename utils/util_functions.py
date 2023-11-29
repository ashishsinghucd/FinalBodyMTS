import os
import shutil
import sys
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_unique_list_of_files(full_coordinates_path, ext_len):
    """
    Function to read all the filenames in a list
    """
    try:
        coordinates_files_list = os.listdir(full_coordinates_path)
        coordinates_files_list = [f[:-ext_len] for f in coordinates_files_list if not f.startswith(".")]
        unique_coordinates_files_list = list(set(coordinates_files_list))
    except Exception as e:
        logger.info("Error in getting the list of the files from: {} {}".format(full_coordinates_path, str(e)))
        unique_coordinates_files_list = None
    return unique_coordinates_files_list


def create_directory_if_not_exists(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except Exception as e:
        logger.info("Error creating the directory: {} {}".format(path, str(e)))


def delete_directory_if_exists(path):
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)

import os
import sys
import logging
import math
import shutil

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
SEED_VAL = 1899797
pid_list = ['P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P2', 'P20', 'P21', 'P22', 'P23',
            'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35', 'P36', 'P37', 'P38',
            'P39', 'P4', 'P40', 'P41', 'P42', 'P43', 'P44', 'P45', 'P46', 'P47', 'P48', 'P49', 'P5', 'P50', 'P51',
            'P52', 'P53', 'P54', 'P55', 'P6', 'P7', 'P8', 'P9']


def split_pids(current_pids_list, seed_value, split_ratio=0.3):
    """
    Function to split the pids into two arrays
    """
    list_pids = current_pids_list
    list_pids.sort()
    logger.info("Current pids are: {}".format(str(current_pids_list)))
    person_count_test = math.floor(split_ratio * len(list_pids))
    person_count_train = len(list_pids) - person_count_test
    logger.info("Total persons in training: {}, testing/validation: {}".format(person_count_train, person_count_test))
    seed_value = int(seed_value)
    np.random.seed(seed_value)
    train_pids = np.random.choice(list_pids, person_count_train, replace=False)
    test_pids = np.array(list(set(list_pids) - set(train_pids)))
    return train_pids, test_pids


def create_format(path):
    train_pids, test_pids = split_pids(pid_list, SEED_VAL)
    train_info = []
    test_info = []

    for root, dirs, files in os.walk(path):
        for f in files:
            full_path = os.path.join(root, f)
            pid = f.split("_", 1)[0]
            exercise_type = f[:-4].split("_")[-1].strip()
            rec = {"video_name": f, "tag": exercise_type}
            if pid in train_pids:
                train_info.append(rec)
                destination_path = os.path.join(output_path, "train", f)
            else:
                test_info.append(rec)
                destination_path = os.path.join(output_path, "test", f)
            shutil.copy(full_path, destination_path)
        #     break
        # break

    test_df = pd.DataFrame(test_info)
    train_df = pd.DataFrame(train_info)

    train_df.to_csv(os.path.join(output_path, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_path, "test.csv"), index=False)


path = "/home/ashish/Results/Datasets/HPE2/FrameToVideos/MP"
output_path = "/home/ashish/Results/Datasets/VP/SOTAFormat/New1/First/MP"

# if not os.path.exists(output_path):
#     os.makedirs(output_path)

create_format(path)

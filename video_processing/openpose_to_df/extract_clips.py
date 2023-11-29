import configparser
import argparse
from pathlib import Path
import os
import rich

import pandas as pd

from video_utils import clip

metadata_path = "./timeslots_55_people.csv"
exercise = "MP"
crop_region = "440,140,860,600"
exercise_types = "N,A,R,Arch".split(",")  # N,A,Ext,R,RB    N,A,R,Arch   N,A,R,BB,Arch
# data_folder = "/home/people/19205522/scratch/GoogleDriveVideos/RawVideos"
# clip_destination = "/home/people/19205522/scratch/Results/Datasets/HPE2/Clips/Row"
data_folder = "/media/ashish/ce461495-3779-4a09-a9db-ef8f5e4c0492/Data/GoogleDriveVideos/RawVideos"  # "/home/ashish"
clip_destination = "/media/ashish/ce461495-3779-4a09-a9db-ef8f5e4c0492/HPE3/SideVideos/MP/Clips"

metadata = pd.read_csv(metadata_path)
metadata = metadata[metadata.Exercise == exercise]
subjects = metadata.P.unique()

crop_region = [int(x) for x in crop_region.split(',')]

for subject in subjects:
    target_rows = metadata[metadata.P == subject]
    # if subject not in [26, 44]:
    #     continue
    for exercise_type in exercise_types:

        entry = target_rows[target_rows.Type == exercise_type]
        if len(entry) == 0:
            rich.print(f'No entry for {subject}, {exercise_type}')
            continue

        rich.print(f'{subject}, {exercise_type}...')
        entry = entry.iloc[0]

        raw_video_name = entry['Video Name']
        if "Frontal" in raw_video_name:
            updated_raw_video_name = raw_video_name.replace("Frontal", "Saggital")
        else:
            updated_raw_video_name = raw_video_name.replace("Front", "Sag")

        video_source = Path(data_folder, updated_raw_video_name + '.MP4')

        if not os.path.exists(video_source):
            rich.print(f'Path does not exist {video_source}')
            continue

        clip_name = updated_raw_video_name + f'_{exercise_type}' '.MP4'
        clip_name = clip_name.replace(' ', '_')

        clip_dst = Path(clip_destination).expanduser()
        clip_dst = Path(clip_dst, clip_name)

        clip_start = int(entry['Time Start (s)'])
        clip_end = int(entry['Time Stop (s)'])

        # if os.path.exists(clip_dst):
        #     continue
        clip(str(video_source), str(clip_dst), (clip_start, clip_end), crop_region)


"""
Row: Ideally 5 types, total 54 persons
3,4,5,6: Total 6 types N, A, R, Ext, Ext 2, RB
43: only 1 which is N
Total clips should be: 54 * 5 + 4 - 4 = 270

BP: Ideally 5 types, total 53 persons
PID 42 is missing
43: only 1 which is N
PID 26: 9 classes, data handling mistake
PID 11 and 24: BB is missing
Total clips should be: 53 * 5 - 1 - 1 - 4 = 259

"""
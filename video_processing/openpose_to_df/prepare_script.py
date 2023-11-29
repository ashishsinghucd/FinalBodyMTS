import configparser
import argparse
from pathlib import Path
import os
import rich

import pandas as pd

from src.utils import clip

config = configparser.ConfigParser()
config.read('config.ini')

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--no-clips', action='store_true')
parser.add_argument('-q', '--quiet', action='store_true')

args = parser.parse_args()

script_path = config['global']['script_path']
exercise = config['global']['exercise']
exercise_types = config['global']['types'].split(',')

metadata_path = config['videos']['metadata_path']
data_folder = config['videos']['data_folder']

keypoint_destination = Path(config['videos']['keypoint_destination']).expanduser()
clip_destination = Path(config['videos']['clip_destination']).expanduser()

openpose_base_path = config['openpose']['openpose_folder']
openpose_bin_path = config['openpose']['openpose_bin_path']

model = config['openpose']['model']

metadata = pd.read_csv(metadata_path)
metadata = metadata[metadata.Exercise == exercise]

# frame_rate = 29.97

crop_region = config['videos']['clip_region']
crop_region = [int(x) for x in crop_region.split(',')]

subjects = metadata.P.unique()

# keypoint_destination.mkdir(exist_ok=True)
clip_destination.mkdir(parents=True, exist_ok=True)

if os.path.exists(script_path):
    os.remove(script_path)

start = ["start=`date +%s`"]
with open(script_path, 'a') as f:
    f.write('; '.join(start) + '\n')

for subject in subjects:
    target_rows = metadata[metadata.P == subject]

    for exercise_type in exercise_types:
        cmd = [f'cd {openpose_base_path}']

        entry = target_rows[target_rows.Type == exercise_type]
        if len(entry) == 0:
            rich.print(f'No entry for {subject}, {exercise_type}')
            continue

        rich.print(f'{subject}, {exercise_type}...')
        entry = entry.iloc[0]

        cmd.append(f'echo {subject}, {exercise_type}')
        op_cmd = f'./{openpose_bin_path}'

        video_source = Path(data_folder,
                            f'P{subject}',
                            entry['Video Name'] + '.MP4')

        clip_name = entry['Video Name'] + f'_{exercise_type}' '.MP4'
        clip_name = clip_name.replace(' ', '_')

        clip_dst = Path(clip_destination).expanduser()
        clip_dst = Path(clip_dst, clip_name)

        clip_start = int(entry['Time Start (s)'])
        clip_end = int(entry['Time Stop (s)'])

        if args.no_clips is False:
            clip(str(video_source),
                 str(clip_dst), (clip_start, clip_end), crop_region)

        series_dst = Path(keypoint_destination, f'P{subject}', exercise_type)
        video_dst = Path(series_dst, 'video.avi')

        series_dst.mkdir(exist_ok=True, parents=True)

        op_cmd += f' --video {clip_dst}'
        op_cmd += f' --model_pose {model}'
        op_cmd += f' --write_json {series_dst}'
        op_cmd += f' --write_video {video_dst}'
        op_cmd += ' --number_people_max 1'

        if args.quiet:
            op_cmd += ' --display 0'

        cmd.append(op_cmd)

        with open(script_path, 'a') as f:
            f.write('; '.join(cmd) + '\n')

end = ["end=`date +%s`", "runtime=$((end-start))", "hours=$((runtime / 3600))",
       "minutes=$(( (runtime % 3600) / 60 ))", "seconds=$(( (runtime % 3600) % 60 ))",
       "echo \"Runtime: $hours:$minutes:$seconds (hh:mm:ss)\" "]

with open(script_path, 'a') as f:
    f.write('; '.join(end) + '\n')

"""
/opt/python/python-3.7.3/bin/python3

./build/examples/openpose/openpose.bin --video /home/ashish/Results/Datasets/HPE/Clips/MP/P2_Frontal_3_A.MP4 --model_pose BODY_25 --write_video /home/ashish/video.avi --number_people_max 1

Openpose on the gray took 1 Hour, 40 Minutes

Openpose on the crf 1 Hour, 41 Minutes and 12 Seconds

Openpopse Runtime: 1:41:44 (hh:mm:ss) on half resolution

one third resolution Runtime: 1:15:56 (hh:mm:ss)

crf 40 openpose Runtime: 1:40:56 (hh:mm:ss)

python prepare_script.py --no-clips --quiet


N,A,Ext,R,RB
N,A,R,Arch

"""

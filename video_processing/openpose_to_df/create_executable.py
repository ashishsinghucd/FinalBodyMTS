import configparser
import argparse
from pathlib import Path
import os

# N,A,Ext,R,RB    N,A,R,Arch   N,A,R,BB,Arch

config = configparser.ConfigParser()
config.read('config.ini')

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--no-clips', action='store_true')
parser.add_argument('-q', '--quiet', action='store_true')

args = parser.parse_args()

script_path = config['global']['script_path']
exercise = config['global']['exercise']
exercise_types = config['global']['types'].split(',')

keypoint_destination = Path(config['videos']['keypoint_destination']).expanduser()
video_source = Path(config['videos']['data_folder']).expanduser()

openpose_base_path = config['openpose']['openpose_folder']
openpose_bin_path = config['openpose']['openpose_bin_path']
model = config['openpose']['model']

if os.path.exists(script_path):
    os.remove(script_path)

start = ["start=`date +%s`"]
with open(script_path, 'a') as f:
    f.write('; '.join(start) + '\n')

list_files = os.listdir(video_source)
list_files = [f for f in list_files if not f.startswith(".")]

for f in list_files:
    cmd = [f'cd {openpose_base_path}']
    subject, exercise_type = f[:-4].split("_")[0], f[:-4].split("_")[-1]
    cmd.append(f'echo {subject}, {exercise_type}')
    op_cmd = f'./{openpose_bin_path}'

    clip_src = Path(video_source, f)

    series_dst = Path(keypoint_destination, f'{subject}', exercise_type)
    video_dst = Path(series_dst, 'video.avi')

    series_dst.mkdir(exist_ok=True, parents=True)

    op_cmd += f' --video {clip_src}'
    op_cmd += f' --model_pose {model}'
    op_cmd += f' --write_json {series_dst}'
    op_cmd += f' --write_video {video_dst}'
    op_cmd += ' --number_people_max 1'
    op_cmd += ' --net_resolution "-1x720"'
    # op_cmd += ' --scale_number 4'
    # op_cmd += ' --scale_gap 0.25'
    op_cmd += ' --display 0'

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

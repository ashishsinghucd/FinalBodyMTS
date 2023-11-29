import pathlib
import configparser
import json

import pandas as pd

from rich.console import Console

from src.utils import aggregate_all_keypoints
from src.utils import get_series_names


console = Console()
config = configparser.ConfigParser()
config.read('config.ini')

model = config['openpose']['model']
keypoints = 25 if model == 'BODY_25' else 18

keypoint_folder = config['videos']['keypoint_destination']
series_destination = config['series']['series_destination']

kp_folders = pathlib.Path(keypoint_folder).expanduser().glob('*/*')
series_destination = pathlib.Path(series_destination).expanduser()
names = get_series_names(model)

series_destination.mkdir(parents=True, exist_ok=True)

mt = {'X': 0, 'Y': 1, 'prob': 2}

for f in kp_folders:
    name = f'{f.parent.name}_{f.name}'

    try:
        points = aggregate_all_keypoints(f, keypoints)
        df = pd.DataFrame(
            columns=[f'{names[i]}_{c}' for i in range(keypoints) for c in
            ['X', 'Y', 'prob']])

        for i in range(keypoints):
            for c in ['X', 'Y', 'prob']:
                df[f'{names[i]}_{c}'] = [x[mt[c]] for x in points[i]]

        df.to_csv(pathlib.Path(series_destination, f'{name}.csv'), index=False)
    except IndexError:
        console.print(f'Exception aggregating {name}', style='red bold')

import json
import pathlib

from moviepy.editor import VideoFileClip
from moviepy.video.fx.all import crop
import numpy as np
import peakutils
from scipy.spatial import distance
from scipy.stats import variation
from scipy.signal import savgol_filter, argrelextrema, find_peaks
import matplotlib.pyplot as plt


def clip(source, destination, time_region, crop_region, save_audio=False):
    """Clip a vide

    Use this method to generate a small subclips from a video. You have to
    provide the source of the video, the destination, and both the time and
    space regions of the clipping.

    Arguments
    ---------
    source: str
        The path of the original video.
    destination: str
        The destination of the final clip.
    time_region: tuple
        The start and end times of the clip, in the format of (start, end).
    crop_region: tuple
        The cropping region of the clip, in the format (x1, y1, x2, y2).
    save_audio: bool
        If true, the audio will be included in the clip, no otherwise.
    """
    video = VideoFileClip(source).subclip(time_region[0], time_region[1])
    cropped = crop(video,
                   x1=crop_region[0], y1=crop_region[1],
                   x2=crop_region[2], y2=crop_region[3])

    video.write_videofile(destination, audio=save_audio,
                          verbose=False, logger=None)


def extract_keypoints(kp_file):
    """Extract keypoints from a keypoint file

    This method will aggregate keypoints into a dictionary. For each keypoint,
    identified by its positional index in the dictionary, three values are
    produced: the x coordinate, the y coordinate, and the probability. The
    final dictionary that is produced will look like this:

    {
        0: (x, y, p), <--- this would be the head in coco
        1: (x, y, p), <--- this would be the body in coco
        ...
    }

    Arguments
    ---------
    kp_file: str
        The path of the keypoint file to read.

    Returns
    -------
    dict
        A dictionary where each key represents a bodypart, and the value is a
        tuple with three values: x, y, and probability.
    """
    points = dict()

    with open(kp_file, 'r') as f:
        data = json.load(f)

    subject = data['people'][0]
    kp = subject['pose_keypoints_2d']

    for point_index, point in enumerate(zip(kp[::3], kp[1::3], kp[2::3])):
        points[point_index] = point

    return points


def aggregate_all_keypoints(kp_folder, expected_kp):
    """Collect all keypoints from a keypoint folder

    This method receives a folder where all keypoint files are stored, and
    returns the aggregated series. The produced dictionary has the keypoints as
    keys, and the values are list of tuples, where each tuple contains the x
    coordinate, the y coordinate, and the probability value.

    Arguments
    ---------
    kp_folder: str
        The folder that contains all the keypoint values.
    expected_kp: int
        The number of expected keypoints. This may be 18 for COCO, 25 for
        Body25. As it might not be needed, it is still good to have it as a
        form of sanity check.

    Returns
    -------
    dict
        A dictionary where the keys are the body parts, and the values are
        series of aggregated (x, y) coordinates and probability values.
    """
    kp_files = list(pathlib.Path(kp_folder).glob('*keypoints.json'))

    # this might not be the best way of sorting kp files, but it works atm
    kp_files.sort()

    all_points = {i: [] for i in range(expected_kp)}

    for f in kp_files:
        points = extract_keypoints(f)

        for i in range(expected_kp):
            all_points[i].append(points[i])

    return all_points


def find_deviation(data, bp="RWrist", axis=1):
    person_deviation = {}
    max_len = -99999
    for person in data:
        max_len = max(max_len, len(data[person][bp]))
    print("Max len:", max_len)
    for person in data:
        if len(data[person][bp]) < (max_len / 2):
            person_deviation[person] = np.nan
    left_body_parts = ["LShoulder", "LHip", "LAnkle", "LSmallToe"]
    for person in data:
        if person in person_deviation and person_deviation[person] is np.nan:
            continue
        for lpb in left_body_parts:
            coordinates = [c[axis] for c in data[person][lpb]]
            is_all_zero = not np.any(coordinates)
            if is_all_zero:
                person_deviation[person] = np.nan
                break

    for person in data:
        if person in person_deviation and person_deviation[person] is np.nan:
            continue
        coordinates = [c[axis] for c in data[person][bp]]
        # person_deviation[person] = variation(coordinates)
        smooth_coordinates = smooth_coordinates_sf(coordinates)
        dst = distance.euclidean(smooth_coordinates, [0] * len(smooth_coordinates))
        person_deviation[person] = dst
    return person_deviation


def aggregate_all_keypoints_multiple(kp_folder, model):
    """Collect all keypoints from a keypoint folder

    This method receives a folder where all keypoint files are stored, and
    returns the aggregated series. The produced dictionary has the keypoints as
    keys, and the values are list of tuples, where each tuple contains the x
    coordinate, the y coordinate, and the probability value.

    Arguments
    ---------
    kp_folder: str
        The folder that contains all the keypoint values.
    model: int
        The number of expected keypoints. This may be 18 for COCO, 25 for
        Body25. As it might not be needed, it is still good to have it as a
        form of sanity check.

    Returns
    -------
    dict
        A dictionary where the keys are the body parts, and the values are
        series of aggregated (x, y) coordinates and probability values.
    """
    kp_files = list(pathlib.Path(kp_folder).glob('*keypoints.json'))
    print("Folder name: {}".format(kp_folder))
    # this might not be the best way of sorting kp files, but it works atm
    kp_files.sort()
    names = get_series_names(model)
    person_data = {}
    setup_flag = False
    total_persons = 7
    for kp_file in kp_files:
        with open(kp_file, 'r') as f:
            data = json.load(f)

        if not setup_flag:
            # TODO: Currently hard coding the number of persons, as this changes whether OpenPose is able to detect
            # TODO: or not
            for p in range(total_persons):
                person_data["P{}".format(p)] = {}
                for bp in names.values():
                    person_data["P{}".format(p)]["{}".format(bp)] = []
            setup_flag = True

        for pid, person in enumerate(data["people"]):
            kp = person['pose_keypoints_2d']
            for point_index, point in enumerate(zip(kp[::3], kp[1::3], kp[2::3])):
                person_data["P{}".format(pid)][names[point_index]].append(point)

    print("Length of each person")
    for p in range(total_persons):
        print("P{}:".format(p), len(person_data["P{}".format(p)]["{}".format(names[0])]))

    # person_deviation = find_deviation(person_data)
    # print("Person deviation")
    # print(person_deviation)
    # person_deviation_filtered = {k: v for k, v in person_deviation.items() if v is not np.nan}
    # print(person_deviation_filtered)
    # pid_max_deviation = min(person_deviation_filtered, key=person_deviation_filtered.get)
    # print(pid_max_deviation)
    # with open("/tmp/jj.json", "w") as fp:
    #     json.dump(person_data, fp)
    return person_data["P0"]


def get_series_names(model):
    return {
        0: "Nose",
        1: "Neck",
        2: "RShoulder",
        3: "RElbow",
        4: "RWrist",
        5: "LShoulder",
        6: "LElbow",
        7: "LWrist",
        8: "RHip",
        9: "RKnee",
        10: "RAnkle",
        11: "LHip",
        12: "LKnee",
        13: "LAnkle",
        14: "REye",
        15: "LEye",
        16: "REar",
        17: "LEar"
    } if model == 'COCO' else {
        0: "Nose",
        1: "Neck",
        2: "RShoulder",
        3: "RElbow",
        4: "RWrist",
        5: "LShoulder",
        6: "LElbow",
        7: "LWrist",
        8: "MidHip",
        9: "RHip",
        10: "RKnee",
        11: "RAnkle",
        12: "LHip",
        13: "LKnee",
        14: "LAnkle",
        15: "REye",
        16: "LEye",
        17: "REar",
        18: "LEar",
        19: "LBigToe",
        20: "LSmallToe",
        21: "LHeel",
        22: "RBigToe",
        23: "RSmallToe",
        24: "RHeel",
        25: "Background"
    }


def smooth_coordinates_sf(raw_coordinates):
    if len(raw_coordinates) < 31:
        return raw_coordinates
    smooth_coordinates = savgol_filter(raw_coordinates, 31, 3)
    # baseline = peakutils.baseline(smooth_coordinates, 5)
    # scaled = smooth_coordinates - baseline
    return smooth_coordinates

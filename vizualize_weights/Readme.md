The code in this folder `vizualize_weights` is used to generate the CAM for interpretation using the deep learning methods such as FCN or
resnet. Currently, it generates a single vector having length same as the original time series. After normalization,
the discriminative region is mapped back to the frames in the original video.

```
INPUT_DATA_PATH = path to data
BASE_PATH= base path
EXTRACTED_FRAMES_PATH = extracted frames of each pid
EXERCISE = type of exercise
SEED_VALUE = split seed value
COMBINATION = folder path
MODEL_NAME = model name fcn or resnet
INDEX_SAMPLE = 9
COUNT_DISPLAY = 1
INTERPRET_CLASS_WEIGHTS = to get an idea about the discriminative region of the model for train data
PID_LIST = list of pids to generate CAM for such P32, P34 etc.
TOP_K = top number of frames to display for
DATA_TYPE = type of the data
MODEL_PATH = model path
```
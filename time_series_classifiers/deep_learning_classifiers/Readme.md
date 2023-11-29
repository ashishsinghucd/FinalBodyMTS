The code in this directory is used to run the deep learning models for multivariate time series classification.
It expects the data in the numpy format. This code has been taken from this link https://github.com/hfawaz/dl-4-tsc. 
Please refer to it for more details. We mainly used fcn and resent models because of their performance. The main
file is the `run_dl.py` to run the deep learning classification.

It expects the following arguments

```
INPUT_DATA_PATH = input data pth
BASE_PATH= base path
EXERCISE = type of exercise
CLASSIFIER_TYPE = classifier type such as fcn or resent
EPOCHS = 600
MULTICLASS_CLASSIFICATION = True
MULTICLASS_DIR = directory path
OUTPUT_PATH =  output path
DATA_TYPE = data type
SEED_VALUES= seed values for different splits
```
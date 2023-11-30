The code in this file `create_train_test_split` takes the output from the file `preprocess_coordinates_new.py` and create the final
format for the train/test to be used for the classification. It uses the config file `train_test_config` for passing
arguments. 

It supports the creating the final format for multivariate time series data in the numpy format to be used for deep
learning methods, sktime format for ROCKET classifiers and table format (this one has not been used for a long time).
It accepts the following arguments.
```
STANDARDIZATION= to standarduze each time individual time series 
SCALING_TYPE= type of standardization
INTERPOLATION= resample the time series to have equal length
VALIDATION_DATA= boolean Flag to create validation data from train data
PADDING= when resampling whether to pad with zero values
SEED_VALUES= to create the number of splits based on seed values
BASE_PATH= path information
INPUT_DATA_PATH= output directory from the preprocess_coordinates_new.py file
EXERCISE= type of exercise
SPLIT_RATIO= split ratio for train/test
SPLIT_VAL_RATIO= split validation ration for train/validation
SEGMENT_STATS_DIR= to store the segment information for additional stats
TRAIN_TEST_DIR= output directory
MULTICLASS_DIR=MulticlassSplit
MIN_LENGTH_TIME_SERIES= minimum length of time series, time series below this length will be dropped
NO_OF_CLASSES=2
DATA_TYPE= type of data type for saving, handy to create wihout overwriting previous types
REVERSE_CLIP= whether to reverse the time series or not
MAX_LENGTH= Max length of time series for resampling to the same length
GENDER_INFO= gender stats file
MAX_LENGTH=default length for resampling
DROP_PARTS= body parts to be dropped, can be used to generate train/test for a single body part
```
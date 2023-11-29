The code in the file `rocket.py` is used to classify the generated train/test data from previous section. It loops
over 3 seeds values and outputs the classification results. 

```
INPUT_DATA_PATH = input data path
BASE_PATH= base path
EXERCISE = type of exercise
SEED_VALUES= seed values for the number of splits
OUTPUT_PATH =  output path
DATA_TYPE = data type
BINARY_CLASSIFICATION = can be used to make binary classification
SEGMENT_STATS_FILE = Results/Datasets/HPE2/SegmentStats/MP/series_body25.csv
GENDER_INFO= demographic information file
SAVE_MODEL= to save model or not
```

The code of this file can be modified for other ROCKET based transforms such as MiniROCKET or MultiROCKET.
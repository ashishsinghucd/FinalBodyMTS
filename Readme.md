This repository contains the code to generate train/test data, run the classification using the ROCKET or deep learning
classifiers and generate CAM for visualization purpose.

Folders and their functionalities

1. `data_processing/create_segments`: used to create the segmentation using the peaks information. The main file is
`preprocess_coordinates_new.py` used to store the peak information in the dataframe.
2. `data_processing/create_train_test_data`: used to create the final train/test/val split for classification. 
3. `time_series_classifiers/deep_learning_classifiers`: used to run the deep learning models such as fcn or resnet to 
classify the data.
4. `time_series_classifiers/time_series_classifiers`: used to run the ROCKET based models to classify the data.
5. `utils`: contains basic utilities functions.
6. `video_processing`: contains scripts to alter the video properties, code to process and merge frames and aggregate
the final data from OpenPose output. 
7. `data_info`: contains files which have demographic info, peaks info for MP and Rowing exercises.
8. `vizualize_weights`: contains the code to interpret the model using the CAM.
## Pose Estimation and Time Series Classification Methods for Efficient Video-Based Exercise Classification

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



Please refer to these publications for more details:
```
Singh, A., Le, B.T., Nguyen, T.L., Whelan, D., O’Reilly, M., Caulfield, B. and Ifrim, G., 2021, February. 
Interpretable classification of human exercise videos through pose estimation and multivariate time series analysis. 
In International Workshop on Health Intelligence (pp. 181-199). Cham: Springer International Publishing.
https://doi.org/10.1007/978-3-030-93080-6_14

Singh, A., Bevilacqua, A., Nguyen, T.L., Hu, F., McGuinness, K., O’Reilly, M., Whelan, D., Caulfield, B. and Ifrim, 
G., 2023. Fast and robust video-based exercise classification via body pose tracking and scalable multivariate 
time series classifiers. Data Mining and Knowledge Discovery, 37(2), pp.873-912.
https://doi.org/10.1007/s10618-022-00895-4

Singh, A., Bevilacqua, A., Aderinola, T.B., Nguyen, T.L., Whelan, D., O’Reilly, M., Caulfield, B. and Ifrim, G., 2023, 
September. An Examination of Wearable Sensors and Video Data Capture for Human Exercise Classification. 
In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (pp. 312-329). Cham: Springer 
Nature Switzerland.
https://doi.org/10.1007/978-3-031-43427-3_19
```
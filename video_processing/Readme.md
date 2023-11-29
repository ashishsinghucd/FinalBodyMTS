The code in the folder `generate_video_formats` is used to modify the properties of a video such as bit rate and
resolution. CRF property is used to modify the bit rate. 

The code in `openpose_to_df` is used to generate the aggregated data from the OpenPose after running on a single video
clip file. The coordinates obtained from each frame are merged and saved into a dataframe. 

The files `convert_frames_video.py` and `extract_frames_video.py` are used to convert frames into a video and extract
frames from a video respectively. 


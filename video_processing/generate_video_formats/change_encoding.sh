#!/usr/bin/env bash

base_path=/home/ashish/Results/Datasets/HPE2
start=`date +%s`


for f in $base_path/FrameToVideos1/MP/*.mp4; do
  echo "Processing" $f
#  echo "$output_path/$f"
  name=${f##*/}
echo "name" $name

ffmpeg -i $base_path/FrameToVideos1/MP/$name -vcodec libx264  $base_path/FrameToVideos2/MP/$name

done

end=`date +%s`
runtime=$((end-start))
hours=$((runtime / 3600)); minutes=$(( (runtime % 3600) / 60 )); seconds=$(( (runtime % 3600) % 60 ));

echo "Runtime: $hours:$minutes:$seconds (hh:mm:ss)"


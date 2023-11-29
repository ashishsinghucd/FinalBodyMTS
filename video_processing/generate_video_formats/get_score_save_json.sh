#!/usr/bin/env bash

cd /home/ashish/Downloads/ffmpeg-git-20210501-amd64-static/

data_path_orig=Clips
data_path_modified=Clips

base_path=/home/ashish/Results/Datasets/HPE2
start=`date +%s`

for f in $base_path/$data_path_orig/MP/*.MP4; do
  echo "Processing" $f
#  echo "$output_path/$f"
  name=${f##*/}
echo "name" $name

 if [ ! -f ./$data_path_modified/$name.json ]; then
      echo "File not found!"
  ./ffmpeg -i $base_path/$data_path_modified/MP/$name -i $base_path/$data_path_orig/MP/$name -lavfi libvmaf="model_path=./model/vmaf_v0.6.1.pkl:log_fmt=json:psnr=1:ssim=1:ssim=1:log_path=harm_mean.json:pool=harmonic_mean" -f null -; mv harm_mean.json  ./default/$name.json

  fi
done

end=`date +%s`
runtime=$((end-start))
hours=$((runtime / 3600)); minutes=$(( (runtime % 3600) / 60 )); seconds=$(( (runtime % 3600) % 60 ));

echo "Runtime: $hours:$minutes:$seconds (hh:mm:ss)"

# first comes the modified then the reference


#for folder in $folders_list; do
#    mkdir -p $folder/MP_Video
#    mkdir -p $folder/Row
#done

#Runtime: 0:17:58 (hh:mm:ss)
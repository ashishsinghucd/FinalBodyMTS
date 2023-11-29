#!/usr/bin/env bash

cd /home/ashish/Downloads/ffmpeg-git-20210501-amd64-static/

data_path_orig=Clips
data_path_modified=CRF16

base_path=/home/ashish/Results/Datasets/HPE2


for f in $base_path/$data_path_orig/MP/*.MP4; do
  echo "Processing" $f
#  echo "$output_path/$f"
  name=${f##*/}
echo "name" $name

./ffmpeg -i $base_path/$data_path_modified/MP/$name -i $base_path/$data_path_orig/MP/$name -lavfi libvmaf="model_path=./model/vmaf_v0.6.1.pkl:log_fmt=json:psnr=1:ssim=1:ssim=1:log_path=harm_mean.json:pool=harmonic_mean" -f null -; jq '."VMAF score", ."PSNR score", ."SSIM score"' harm_mean.json | tr '\n' ',' >> one.txt ; echo $name >> one.txt

done

# first comes the modified then the reference


#for folder in $folders_list; do
#    mkdir -p $folder/MP_Video
#    mkdir -p $folder/Row
#done
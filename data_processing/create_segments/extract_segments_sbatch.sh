#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ashish.singh@ucdconnect.ie
#SBATCH --job-name=extract_segments_from_clips

#SBATCH -o /scratch/%u/%x-%N-%j.out    # Output file
#SBATCH -e /scratch/%u/%x-%N-%j.err    # Error file

export EXTRACT_SEGMENT_DIR=/home/people/19205522/Research/Codes/human_pose_estimation/extract_segments
export PYTHONPATH=$PYTHONPATH:/home/people/19205522/Research/Codes/human_pose_estimation/

cd $EXTRACT_SEGMENT_DIR

module load anaconda/3.5.2.0
conda activate /home/people/19205522/.conda/envs/mlutils/

time python preprocess_coordinates.py /home/people/19205522/scratch/KerasRealtime MP False False 5
date;


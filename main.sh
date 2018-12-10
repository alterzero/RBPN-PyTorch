#!/bin/bash

source /home/mharis/virtualenv/myenv_3/bin/activate

export CUDA_VISIBLE_DEVICES=$3

python main.py --nFrames $1 --upscale_factor $2 --model_type $4| tee -a log/$1_$2_$4_$(hostname)_$(date "+%d_%m_%y_%H_%M_%S").log 


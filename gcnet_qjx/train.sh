#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py --mode train --model_name model_test \
--data_path /media/harddriver/stc/SR_stereo/stereo/monodepth-master/data/ \
--filenames_file ./filenames/list.txt \
--log_directory ./log2/ \
--input_height 256 \
--input_width 512 \
--batch_size 1 \
--num_gpus 1 \
--num_epochs 5 \
--max_disparity 192

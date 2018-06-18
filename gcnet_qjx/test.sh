#!/usr/bin/env bash
python main.py --mode test --model_name 0429 \
--data_path /cluster/scratch/maxh/dataset/ \
--filenames_file ./prediction_files/images.txt \
--log_directory ./log/ \
--output_directory /cluster/scratch/maxh/dataset/ \
--batch_size 1 \
--num_gpus 1 \
--num_epochs 1 \
--max_disparity 192

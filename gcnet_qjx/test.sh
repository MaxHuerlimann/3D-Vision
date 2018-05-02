#!/usr/bin/env bash
python main.py --mode test --model_name model_test \
--data_path ./testingdata/ \
--filenames_file ./test_files.txt \
--log_directory ./log/ \
--output_directory ./log/ \
--batch_size 1 \
--num_gpus 1 \
--num_epochs 1 \
--max_disparity 192

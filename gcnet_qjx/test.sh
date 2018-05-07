#!/usr/bin/env bash
python main.py --mode test --model_name model_test \
--data_path ./testingdata/ \
--filenames_file ./test_files.txt \
--log_directory ./log/ \
--output_directory ./log/ \
--batch_size 1 \
<<<<<<< 7e2e4cd7bafef9abdf785cfd7ec036f55e60d9bc
--num_gpus 1 \
=======
--num_gpus 0 \
>>>>>>> Implement GCNet as the disparity estimator
--num_epochs 1 \
--max_disparity 192

# Stereo-SfMLearner

This code enhances the existing code from the [SfMLearner](https://github.com/tinghuiz/SfMLearner)

[Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz/), [Matthew Brown](http://matthewalunbrown.com/research/research.html), [Noah Snavely](http://www.cs.cornell.edu/~snavely/), [David G. Lowe](http://www.cs.ubc.ca/~lowe/home.html)

The disparity estimation was replaced with an implementation of

Unsupervised Ego-Motion from video

## Prerequisites

These libraries and frameworks were used for this code:
Ubuntu 16.04
python 3.6.5
Tensorflow 1.7.0
CUDA 9.0
cudnn 7.1.2
opencv 3.3.1
scipy 1.0.1
matplotlib 2.2.2
scikit-image 0.13.1

The network was trained on the ETH Leonhard cluster.

## Code Contained
|Folder:|Content:|
|--|--|
|root|Code handling the main processe (main.py and utils.py), definition of the networks, data loading, training and testing|
|--|--|
|root/data/|Code for putting the image files into desired sequences for training, creating text files for intrinsics and lists of the files for data loading later on|
|--|--|
|root/models/|Scripts to download pretrained models|

## Preparing Training Data
To train the network, the training data has to be formatted.

First, download the [KITTI odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) dataset from [here](http://www.cvlibs.net/download.php?file=data_odometry_color.zip).
The disparity maps have to be produced with the GC-Net code. This is done before the actual training to speed up the process, as the network for disparity is not being trained further. It will output 384x1280 maps.
```bash
python main.py --mode test --model_name 0429 --data_path /path/to/dataset/ --filenames_file /path/and/name/to/save/filelist/file/ --log_directory /where/to/save/logs/ --output_directory /path/to/save/depths/
```

Then, sequences are formed out of the dataset images and depth in the same order for training. Any uneven number for the sequence length can be used, but only sequence lengths 3 and 5 have been tested.
```bash
python data/prepare_depth_data.py --dataset_dir=/path/to/dataset/ --depth_dir=/path/to/precalculated/depths/ --dump_root_image=/path/to/save/image/sequences/ --dump_root_depth=/path/to/save/depth/sequences/ --seq_length="desired sequence length" --img_width="desired image width" --img_height="desire image height"
```

## Training

## Testing


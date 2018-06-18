# Stereo-SfMLearner

This code enhances the existing code from the [SfMLearner](https://github.com/tinghuiz/SfMLearner)

[Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz/), [Matthew Brown](http://matthewalunbrown.com/research/research.html), [Noah Snavely](http://www.cs.cornell.edu/~snavely/), [David G. Lowe](http://www.cs.ubc.ca/~lowe/home.html)

The disparity estimation was replaced with an implementation of [GC-Net](https://arxiv.org/pdf/1703.04309.pdf) Alex Kendall et al. The implementation itself was written by Jiaxiong Qiu, based off of an implementation by [Lin Hung Shi](https://github.com/LinHungShi/GCNetwork).

Unsupervised Ego-Motion from video

## Prerequisites
The code was written in python on Ubuntu 16.04

These libraries and frameworks were used for this code:<br/>
python 3.6.5<br/>
Tensorflow 1.7.0<br/>
CUDA 9.0<br/>
cudnn 7.1.2<br/>
opencv 3.3.1<br/>
scipy 1.0.1<br/>
matplotlib 2.2.2<br/>
scikit-image 0.13.1

The network was trained on the ETH Leonhard cluster.

## Code Contained
|Folder:|Content:|
| ------- | -------- |
|StereoSfMLearner|The main code developed by us. Code handling the main processes (main.py and utils.py), definition of the networks, data loading, training and testing|
|root/data/|Code for putting the image files into desired sequences for training, creating text files for intrinsics and lists of the files for data loading later on|
|gcnet|Implementation of the GC-Net|

## Preparing Training Data
To train the network, the training data has to be formatted.

First, download the [KITTI odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) dataset from [here](http://www.cvlibs.net/download.php?file=data_odometry_color.zip).
The disparity maps have to be produced with the GC-Net code. This is done before the actual training to speed up the process, as the network for disparity is not being trained further. It will output 384x1280 maps.
```bash
python main.py --mode test --model_name 0429 --data_path /path/to/dataset/ --filenames_file /path/and/name/to/save/filelist/file/ --log_directory /where/to/save/logs/ --output_directory /path/to/save/depths/
```

Then, sequences are formed out of the dataset images and depth in the same order for training. Any uneven number for the sequence length can be used, but only sequence lengths 3 and 5 have been tested.
```bash
python data/prepare_train_data.py --dataset_dir=/path/to/dataset/ --depth_dir=/path/to/precalculated/depths/ --dump_root_image=/path/to/save/image/sequences/ --dump_root_depth=/path/to/save/depth/sequences/ --seq_length="desired sequence length" --img_width="desired image width" --img_height="desire image height"
```

## Training
To train the code the following command is used:
```bash
python train.py --dataset_dir /path/to/formatted/image/sequences/ --depths_dir /path/to/formatted/depth/sequences/ --checkpoint_dir /path/to/save/checkpoints/ --img_width [image width] --img_height [image height] --batch_size [batch size] --seq_length [sequence length]
```
In the train.py file the parameters for training can adapted.

## Testing
Code from the original implementation can be used to download groundtruth for the poses on the KITTI odometry dataset sequences 9 and 10
```bash
bash ./kitti_eval/download_kitti_pose_eval_data.sh
```
Then, the pretrained model is used to predict the poses of any desire sequence of images
```bash
python test_kitti_pose.py --test_seq [sequence_id] --dataset_dir /path/to/KITTI/odometry/set/ --output_dir /path/to/output/directory/ --posenet_model /path/to/pre-trained/model/file/ --depths_dir /cluster/scratch/maxh/dataset/depths_np/
```
Notice that all the predictions and ground-truth are 5-frame snippets with the format of `timestamp tx ty tz qx qy qz qw` consistent with the [TUM evaluation toolkit](https://vision.in.tum.de/data/datasets/rgbd-dataset/tools#evaluation). Then you could run 
```bash
python kitti_eval/eval_pose.py --gtruth_dir=/directory/of/groundtruth/trajectory/files/ --pred_dir=/directory/of/predicted/trajectory/files/
```

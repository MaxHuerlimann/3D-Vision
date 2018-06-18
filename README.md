# Stereo-SfMLearner

Unsupervised Ego-Motion from video

This code enhances the existing code from the [SfMLearner](https://github.com/tinghuiz/SfMLearner)

[Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz/), [Matthew Brown](http://matthewalunbrown.com/research/research.html), [Noah Snavely](http://www.cs.cornell.edu/~snavely/), [David G. Lowe](http://www.cs.ubc.ca/~lowe/home.html)

The disparity estimation was replaced with an implementation of [GC-Net](https://arxiv.org/pdf/1703.04309.pdf) Alex Kendall et al. The implementation itself was written by Jiaxiong Qiu, based off of an implementation by [Lin Hung Shi](https://github.com/LinHungShi/GCNetwork).

## Prerequisites
The code was written in python on Ubuntu 16.04 and the environment was managed with miniconda.

These libraries and frameworks were used for this code:<br/>
python 3.6.5<br/>
Tensorflow 1.7.0 (pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.7.1-cp36-cp36m-linux_x86_64.whl)<br/>
CUDA 9.0 (following https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)<br/>
cudnn 7.1.2<br/>
opencv 3.3.1 (pip install opencv)<br/>
scipy 1.1.0 (pip install scipy)<br/>
matplotlib 2.2.2 (pip install matplotlib)<br/>
scikit-image 0.13.1 (pip install scikit-image)

The network was trained on the ETH Leonhard cluster.

## Code Contained
There are two different implementations. One is with both gcnet and posenet in one network while the other precalculates the depths for the whole dataset and feeds them to the posenet directly. The implementation with precalculated depths was used mainly as the training is faster and there is more flexibility in training the network (e.g. variable batch size)

|Folder:|Content:|
| ------- | -------- |
|StereoSfMLearner|The main code developed by us combining GC-Net and posnet. Code handling the main processes (main.py and utils.py), definition of the networks, data loading, training and testing|
|StereoSfMLearnerPrecalc|The main code developed by us using precalculated depths. Code handling the main processes (main.py and utils.py), definition of the networks, data loading, training and testing|
|StereoSfMLearner*/data/|Code for putting the image files into desired sequences for training, creating text files for intrinsics and lists of the files for data loading later on|
|StereoSfMLearner*/data/kitti/|The dataloader classes for preparing the training data.|
|gcnet|Implementation of the GC-Net. The most important file is the main file. It gets executed to estimate the disparities. The subfolder prediction_files contains an example file containing the paths to the images, which was created using generate_image_list.py|
|sampleimages|Sample images of the KITTI odometry dataset and corresponding estimated disparity maps|

Almost all of the commands given are specific to the pre-estimated disparity implementation. Adapting for the combined implementation is straightforward as the scripts have the same names, only the arguments have to be changed.

## Preparing Training Data
To train the network, the training data has to be formatted.

First, download the [KITTI odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) dataset from [here](http://www.cvlibs.net/download.php?file=data_odometry_color.zip).
For the posenet only implementation, the disparity maps have to be produced with the GC-Net code beforehand. This is done before the actual training to speed up the process, as the network for disparity is not being trained further and allow bigger batch sizes than 1 (The GC-Net implementation only allows for batch size 1). It will output 384x1280 maps. 
```bash
python gcnet/main.py --mode test --model_name 0429 --data_path /path/to/dataset/ --filenames_file /path/to/file/with/image/paths/ --log_directory /where/to/save/logs/ --output_directory /path/to/save/depths/
```

Then, sequences are formed out of the dataset images and depth in the same order for training. Any uneven number for the sequence length can be used, but only sequence lengths 3 and 5 have been tested.
```bash
python StereoSfMLearnerPrecalc/data/prepare_train_data.py --dataset_dir=/path/to/dataset/ --depth_dir=/path/to/precalculated/depths/ --dump_root_image=/path/to/save/image/sequences/ --dump_root_depth=/path/to/save/depth/sequences/ --seq_length="desired sequence length" --img_width="desired image width" --img_height="desire image height"
```

## Training
To train the code the following command is used:
```bash
python StereoSfMLearnerPrecalc/train.py --dataset_dir /path/to/formatted/image/sequences/ --depths_dir /path/to/formatted/depth/sequences/ --checkpoint_dir /path/to/save/checkpoints/ --img_width [image width] --img_height [image height] --batch_size [batch size] --seq_length [sequence length]
```
In the train.py file the parameters for training can adapted. Generally during training the cost oscillates quite heavily and overfits quickly, so smaller learning rates are adviced.

## Testing
Code from the original implementation can be used to download groundtruth for the poses on the KITTI odometry dataset sequences 9 and 10
```bash
bash StereoSfMLearnerPrecalc/kitti_eval/download_kitti_pose_eval_data.sh
```
Then, the pretrained model is used to predict the poses of any desire sequence of images
```bash
python StereoSfMLearnerPrecalc/test_kitti_pose.py --test_seq [sequence_id] --dataset_dir /path/to/KITTI/odometry/set/ --output_dir /path/to/output/directory/ --posenet_model /path/to/pre-trained/model/file/ --depths_dir /cluster/scratch/maxh/dataset/depths_np/
```
Notice that all the predictions and ground-truth are 5-frame snippets with the format of `timestamp tx ty tz qx qy qz qw` consistent with the [TUM evaluation toolkit](https://vision.in.tum.de/data/datasets/rgbd-dataset/tools#evaluation). Then you could run 
```bash
python StereoSfMLearnerPrecalc/kitti_eval/eval_pose.py --gtruth_dir=/directory/of/groundtruth/trajectory/files/ --pred_dir=/directory/of/predicted/trajectory/files/
```

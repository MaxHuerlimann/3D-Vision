from __future__ import division
import argparse
import scipy.misc
import numpy as np
from glob import glob
from joblib import Parallel, delayed
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, required=True, help="where the dataset is stored")
parser.add_argument("--depth_dir", type=str, required=True, help="where the precalculated depths are stored")
parser.add_argument("--dump_root_image", type=str, required=True, help="Where to dump the image sequences")
parser.add_argument("--dump_root_depth", type=str, required=True, help="Where to dump the depth sequences")
parser.add_argument("--seq_length", type=int, required=True, help="Length of each training sequence")
parser.add_argument("--img_height", type=int, default=128, help="image height")
parser.add_argument("--img_width", type=int, default=416, help="image width")
parser.add_argument("--num_threads", type=int, default=4, help="number of threads to use")
args = parser.parse_args()

def concat_image_seq(seq):
    for i, im in enumerate(seq):
        if i == 0:
            res = im
        else:
            res = np.hstack((res, im))
    return res

def dump_example(n, mode):
    if n % 2000 == 0:
        print('Progress %d/%d....' % (n, data_loader.num_train))
    if mode == 'image':
        example = data_loader.get_train_example_with_idx(n)
        intrinsics = example['intrinsics']
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
    else:
        example = depth_loader.get_train_example_with_idx(n)
    if example == False:
        return
    image_seq = concat_image_seq(example['image_seq'])
    if mode == 'image':
        dump_dir = os.path.join(args.dump_root_image, example['folder_name'])
    else:
        dump_dir = os.path.join(args.dump_root_depth, example['folder_name'])
    # if not os.path.isdir(dump_dir):
    #     os.makedirs(dump_dir, exist_ok=True)
    try: 
        os.makedirs(dump_dir)
    except OSError:
        if not os.path.isdir(dump_dir):
            raise
    if mode == 'image':
        dump_img_file = dump_dir + '/%s.jpg' % example['file_name']
        scipy.misc.imsave(dump_img_file, image_seq.astype(np.uint8))
        dump_cam_file = dump_dir + '/%s_cam.txt' % example['file_name']
        with open(dump_cam_file, 'w') as f:
            f.write('%f,0.,%f,0.,%f,%f,0.,0.,1.' % (fx, cx, fy, cy))
    else:
        dump_img_file = dump_dir + '/%s' % example['file_name']
        np.save(dump_img_file, image_seq)
def main():
    if not os.path.exists(args.dump_root_image):
        os.makedirs(args.dump_root_image)

    if not os.path.exists(args.dump_root_depth):
        os.makedirs(args.dump_root_depth)

    # Data loader for depths
    global depth_loader
    from kitti.kitti_depth_loader import kitti_depth_loader
    depth_loader = kitti_depth_loader(args.depth_dir,
                                    img_height=args.img_height,
                                    img_width=args.img_width,
                                    seq_length=args.seq_length)

    # Data loader for images
    global data_loader
    from kitti.kitti_odom_loader import kitti_odom_loader
    data_loader = kitti_odom_loader(args.dataset_dir,
                                    img_height=args.img_height,
                                    img_width=args.img_width,
                                    seq_length=args.seq_length)


    Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n, 'image') for n in range(data_loader.num_train))
    Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n, 'depth') for n in range(data_loader.num_train))

    # Split into train/val (shut off right now) and write to txt files
    subfolders = os.listdir(args.dump_root_image)
    subfolders.sort()
    np.random.seed(8964)
    with open(args.dump_root_image + '/train.txt', 'w') as tf:
        with open(args.dump_root_image + '/val.txt', 'w') as vf:
            for s in subfolders:
                if not os.path.isdir(args.dump_root_image + '/%s' % s):
                    continue
                imfiles = glob(os.path.join(args.dump_root_image, s, '*.jpg'))
                frame_ids = [os.path.basename(fi).split('.')[0] for fi in imfiles]
                frame_ids.sort()
                for frame in frame_ids:
                    if np.random.random() < 0:
                        vf.write('%s %s\n' % (s, frame))
                    else:
                        tf.write('%s %s\n' % (s, frame))


main()


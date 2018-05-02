#!/usr/bin/python
# -*-coding:utf-8-*-

# Generate absolute path list of images for `FlyingThings3D - Scene Flow Dataset`
# Put `generate_image_list.py` under the same directory with `frame_finalpass` folder and `disparity` folder

import os
ROOT_DIR = os.getcwd()
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def generate_image_list(filepath = "./"):

	classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
	image = [img for img in classes if img.find('frames_cleanpass') > -1]
	disp = [dsp for dsp in classes if dsp.find('disparity') > -1]

	all_left_img = []
	all_right_img = []
	all_left_disp = []

	f_train = open('train2.txt', 'w')


	driving_dir = filepath + [x for x in image][0] + '/'
	driving_disp = filepath + [x for x in disp][0]

	subdir1 = ['15mm_focallength', '15mm_focallength']
	subdir2 = ['scene_backwards', 'scene_forwards']
	subdir3 = ['fast', 'slow']

	for i in subdir1:
		for j in subdir2:
			for k in subdir3:
				imm_l = os.listdir(driving_dir + i + '/' + j + '/' + k + '/left/')
				for im in imm_l:
					if is_image_file(driving_dir + i + '/' + j + '/' + k + '/left/' + im):
						all_left_img.append(driving_dir + i + '/' + j + '/' + k + '/left/' + im)
						#all_left_disp.append(driving_disp + '/' + i + '/' + j + '/' + k + '/left/' + im.split(".")[0] + '.pfm')

					if is_image_file(driving_dir + i + '/' + j + '/' + k + '/right/' + im):
						all_right_img.append(driving_dir + i + '/' + j + '/' + k + '/right/' + im)

	files = os.listdir("./SFdisparity")
	files.sort(key=lambda x:int(x[:-4]))

	train = [img for img in files]
	all_left_disp = [filepath + "SFdisparity/" + img for img in train]

	l = len(all_left_disp)
	#print(l)
	counter = 0

	for data_file1, data_file2, label_file in zip(all_left_img, all_right_img, all_left_disp):
		counter += 1
		line = str(data_file1) + ' ' + str(data_file2) + ' ' + str(label_file)
		if counter < l:
			line += '\n'
		f_train.write(line)

	f_train.close()
	print("Image list generation completed!")


if __name__ == '__main__':
	file_path = ROOT_DIR + '/'
	generate_image_list(file_path)

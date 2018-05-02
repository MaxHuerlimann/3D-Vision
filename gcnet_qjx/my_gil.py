import os
import os.path
ROOT_DIR = os.getcwd()
IMG_EXTENSIONS = [
	'.jpg', '.JPG', '.jpeg', '.JPEG',
	'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader_train(filepath):
	left_fold = 'image_2/'
	right_fold = 'image_3/'
	disp_L = 'disp_occ_0/'
	disp_R = 'disp_occ_1/'

	train = [img for img in os.listdir(filepath + left_fold) if img.find('_10') > -1]

	left_train = [filepath + left_fold + img for img in train]
	right_train = [filepath + right_fold + img for img in train]
	disp_train_L = [filepath + disp_L + img for img in train]
	#disp_train_R = [filepath+disp_R+img for img in train]
	l = len(train)
	f_train = open('train.txt', 'w')

	counter = 0
	for data_file1, data_file2, label_file1 in zip(left_train, right_train, disp_train_L):
		counter = counter + 1
		line = str(data_file1) + ' ' + str(data_file2) + ' ' + str(label_file1)
		if counter < l:
			line = line + '\n'
		f_train.write(line)

	f_train.close()
	print("Image list generation completed!")

def dataloader_test(filepath):
    left_fold = 'image_2/'
    right_fold = 'image_3/'

    test = [img for img in os.listdir(filepath + left_fold) if img.find('_10') > -1]

    left_test = [filepath + left_fold + img for img in test]
    right_test = [filepath + right_fold + img for img in test]

    l = len(test)
    f_test = open('test.txt', 'w')
    #line = str(left_test[0]) + ' ' + str(right_test[0])
    # f_test.write(line)

    counter = 0
    for data_file1, data_file2 in zip(left_test, right_test):
    	counter = counter + 1
    	line = str(data_file1) + ' ' + str(data_file2)
    	if counter < l:
    		line = line + '\n'
    	f_test.write(line)

    f_test.close()
    print("Image list generation completed!")

def main():
	#filepath = os.path.join(ROOT_DIR, 'data_scene_flow/training/')
	# dataloader_train(filepath)

	filepath = os.path.join(ROOT_DIR, 'data_scene_flow/testing/')
	dataloader_test(filepath)

if __name__ == "__main__":
	main()
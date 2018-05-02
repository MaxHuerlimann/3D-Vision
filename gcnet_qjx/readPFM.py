import numpy as np
import re
import sys
from PIL import Image
import cv2
from scipy import misc

def load_pfm(ad):
	with open(ad, 'rb') as file:
		color = None
		width = None
		height = None
		scale = None
		endian = None
		header = file.readline().decode('utf8', 'ignore').rstrip()
		if header == 'PF':
			color = True
		elif header == 'Pf':
			color = False
		else:
			raise Exception('Not a PFM file.')
		dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf8', 'ignore'))
		if dim_match:
			width, height = map(int, dim_match.groups())
		else:
			raise Exception('Malformed PFM header.')
		scale = float(file.readline().decode('utf8', 'ignore').rstrip())
		if scale < 0:  # little-endian
			endian = '<'
			scale = -scale
		else:
			endian = '>'  # big-endian
		img = np.fromfile(file, endian + 'f')
		shape = (height, width, 3) if color else (height, width)
		img = np.reshape(img, shape)
		img = cv2.flip(img, 0)
	file.close()
	return np.array(img)

if __name__ == '__main__':
	counter = 0
	with open('./train.lst', 'rb') as file:
		while True:
			line = file.readline()
			if not line:
				break
			files = line.split()
			img = Image.fromarray(load_pfm(files[2]))
			if img.mode != 'L':
				img = img.convert('L')
			s = "%04d" % counter
			output = "./SFdisparity/" + s + ".png"
			img.save(output)
			counter += 1
			print(counter)
#s = "%04d" % counter
#print(s)

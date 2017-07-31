#!/usr/bin/env python
import sys 
import argparse 
import os.path 
import os

import cv2 

import numpy as np 
import libtiff as lt 
from libtiff import TIFF

from glob import glob 

def main():
	usage = """deepflow_viz.py [input_dir] [ref frame 1 name] <ref frame 2 name> ...

Expects refframes directory in [input_dir] to be already filled with i-frames

Example: 
./deepflow_refframes_viz.py ./register/20160412stk0001-0008/  frame_0001.tif frame_0501.tif frame_1001.tif frame_1501.tif

For help:
./deepflow_refframes_viz.py -h 

Ben Lansdell
12/17/2016
"""

	parser = argparse.ArgumentParser()
	parser.add_argument('path_in', help='input directory with frames already placed in it')
	parser.add_argument('rframes', help='list of global reference frames', nargs = '+', type = str)
	args = parser.parse_args()

	#Test code
	#class Args:
	#	pass 
	#args = Args()
	#args.path_in = './register/20160412stk0001-0008/'
	#args.rframes = ['frame_0001.tif', 'frame_0501.tif', 'frame_1001.tif', 'frame_1501.tif']

	#Get all files in 
	iframes = sorted(glob(args.path_in + 'refframes/*.tif'))
	nF = len(iframes)
	nR = len(args.rframes)

	#Find indices of rframes in iframes list
	rframeidx = [iframes.index(args.path_in + 'refframes/' + rf) for rf in args.rframes]

	#Make image directory
	if not os.path.isdir(args.path_in + 'dm_viz'):
		os.makedirs(args.path_in + 'dm_viz')

	images = []
	for fn_in in iframes:
		image = cv2.imread(fn_in)
		images.append(image)

	ny = images[0].shape[0]
	nx = images[0].shape[1]

	#Vizualize DM results
	for r in range(nR):
		i = rframeidx[r]
		im1 = iframes[i]
		#Get frame number 
		fn1 = int(os.path.splitext(os.path.basename(im1))[0].split('_')[1])
		for j in range(nF):
			#j = rframeidx[s]
			im2 = iframes[j]
			if i != j:
				fn2 = int(os.path.splitext(os.path.basename(im2))[0].split('_')[1])
				print("Drawing DeepMatching results between frame %d and %d" %(fn1, fn2))
				fn_in = args.path_in + 'corrmatrix/%04d_%04d.txt'%(fn1,fn2)
				fn_out = args.path_in + 'dm_viz/%04d_%04d.png'%(fn1,fn2)
				image1 = images[i].copy()
				image2 = images[j].copy()

				#Open file and draw correspondences on each image
				with open(fn_in, 'r') as corr:
					for line in corr.readlines():
						[dmx1,dmy1,dmx2,dmy2] = [int(a) for a in line.split()[0:4]]
						#Compute color
						a = np.complex(dmx1-(nx/2.), (dmy1-(ny/2.)))
						angle = np.angle(a)
						if angle < 0:
							angle += 2*np.pi
						hsv = np.array([[[int(179*angle/(2*np.pi)), 255, 255]]], dtype=np.uint8)
						rgb = tuple(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0,0,:].astype(int))
						#Place point on grid
						cv2.circle(image1, (dmx1,dmy1), 2, rgb) 
						cv2.circle(image2, (dmx2,dmy2), 2, rgb) 

				#Combine images and draw/save
				image = np.hstack((image1,image2))
				cv2.imwrite(fn_out, image)

				#cv2.imshow('deep matching results', image)
				#cv2.waitKey()

if __name__ == "__main__":
	sys.exit(main())

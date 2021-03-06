#!/usr/bin/env python
import sys 
import argparse 
import os.path 
import os

import numpy as np 
import libtiff as lt 
from libtiff import TIFF

from glob import glob 

DM = '~/deepmatching/deep_matching_gpu.py'
DF = '~/deepflow/deepflow2-static'

def main():
	usage = """deepflow_simmatrix.py [input_dir]

Expects refframes directory in [input_dir] to be already filled with i-frames

Example: 
./deepflow_simmatrix_viz.py ./simmatrix/20160412/

For help:
./deepflow_simmatrix_viz.py -h 

Ben Lansdell
01/04/2016
"""

	parser = argparse.ArgumentParser()
	parser.add_argument('path_in', help='input directory with frames already placed in it')
	parser.add_argument('rframes', help='list of global reference frames', nargs = '+', type = str)
	parser.add_argument('-ext', help='extension of file to convert tifs to', default='png', type = str)
	parser.add_argument('-downscale', help='Factor to downscale image sizes by', default = 2, type = int)
	args = parser.parse_args()

	#Test code
	#class Args:
	#	pass 
	#args = Args()
	#args.path_in = './simmatrix/20160412/'
	#args.rframes = ['frame_0001.tif', 'frame_0251.tif']
	#args.ext = 'png'
	#args.downscale = 2 

	#Get all files in 
	iframes = sorted(glob(args.path_in + 'refframes/*.tif'))
	nF = len(iframes)
	nR = len(args.rframes)

	#print iframes 

	ext = args.ext
	downscale = args.downscale

	#Find indices of rframes in iframes list
	rframeidx = [iframes.index(args.path_in + 'refframes/' + rf) for rf in args.rframes]

	#Make video directory
	if not os.path.isdir(args.path_in + 'corrmatrix'):
		os.makedirs(args.path_in + 'corrmatrix')

	images = []
	for fn_in in iframes: # do stuff with image
		# to open a tiff file for reading:
		tif = TIFF.open(fn_in, mode='r')
		image = tif.read_image()
		images.append(image)
		tif.close()

	for frame in iframes:
		im_out = frame[0:-3] + ext
		#print im_out 
		os.system('convert %s -auto-level -depth 8 %s'%(frame, im_out))

	#Run DeepMatching
	for r in range(nR):
		i = rframeidx[r]
		im1 = iframes[i][0:-3] + ext
		#Get frame number
		fn1 = int(os.path.splitext(os.path.basename(im1))[0].split('_')[1])
		for j in range(nF):
			if i != j:
				#im2 = args.path_in + 'refframes/frame_%04d.%s'%(iframes[j], ext)
				im2 = iframes[j][0:-3] + ext
				fn2 = int(os.path.splitext(os.path.basename(im2))[0].split('_')[1])
				print("DeepMatching between frame %d and %d" %(fn1, fn2))
				fn_out = args.path_in + 'corrmatrix/%04d_%04d.txt'%(fn1,fn2)
				os.system('python %s %s %s -ds %d -out %s' %(DM, im1, im2, downscale, fn_out)) 

	#Run DeepFlow
	for r in range(nR):
		i = rframeidx[r]
		im1 = iframes[i]
		im1 = im1[0:-3] + ext
		#Get frame number 
		fn1 = int(os.path.splitext(os.path.basename(im1))[0].split('_')[1])
		for j in range(nF):
			if i != j:
				im2 = iframes[j]
				im2 = im2[0:-3] + ext
				fn2 = int(os.path.splitext(os.path.basename(im2))[0].split('_')[1])
				print("DeepFlow between frame %d and %d" %(fn1, fn2))
				fn_out = args.path_in + 'corrmatrix/%04d_%04d.flo'%(fn1,fn2)
				matches = args.path_in + 'corrmatrix/%04d_%04d.txt'%(fn1,fn2)
				os.system(DF + ' %s %s %s -match %s' %(im1, im2, fn_out, matches)) 

if __name__ == "__main__":
	sys.exit(main())

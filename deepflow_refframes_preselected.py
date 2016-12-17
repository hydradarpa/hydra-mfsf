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
	usage = """deepflow_refframes_preselected.py [input_dir] [ref frame 1 name] <ref frame 2 name> ...

Expects refframes directory in [input_dir] to be already filled with i-frames

Example: 
./deepflow_refframes_preselected.py ./register/20160412stk0001-0008/  frame_0001.tif frame_0501.tif frame_1001.tif frame_1501.tif

For help:
./deepflow_refframes_preselected.py -h 

Ben Lansdell
12/17/2016
"""

	parser = argparse.ArgumentParser()
	parser.add_argument('path_in', help='input directory with frames already placed in it')
	parser.add_argument('rframes', help='list of global reference frames', nargs = '+', type = str)
	args = parser.parse_args()

	#Test code
	class Args:
		pass 
	args = Args()
	args.path_in = './register/20160412stk0001-0008/'
	args.rframes = ['frame_0001.tif', 'frame_0501.tif', 'frame_1001.tif', 'frame_1501.tif']

	#Get all files in 
	iframes = sorted(glob(args.path_in + 'refframes/*'))
	nF = len(iframes)
	nR = len(args.rframes)

	#Find indices of rframes in iframes list
	rframeidx = [iframes.index(args.path_in + 'refframes/' + rf) for rf in args.rframes]

	#Make video directory
	if not os.path.isdir(args.path_in + 'corrmatrix'):
		os.makedirs(args.path_in + 'corrmatrix')
	#if not os.path.isdir(args.dir_out + 'mfsf'):
	#	os.makedirs(args.dir_out + 'mfsf')

	#images = []
	#for fn_in in iframes: # do stuff with image
	#	# to open a tiff file for reading:
	#	tif = TIFF.open(fn_in, mode='r')
	#	image = tif.read_image()
	#	images.append(image)
	#	tif.close()

	#Run DeepMatching
	for r in range(nR):
		i = rframeidx[r]
		im1 = iframes[i]
		#Get frame number 
		fn1 = int(os.path.splitext(os.path.basename(im1))[0].split('_')[1])
		for j in range(nF):
			if i != j:
				im2 = iframes[j]
				fn2 = int(os.path.splitext(os.path.basename(im2))[0].split('_')[1])
				print("DeepMatching between frame %d and %d" %(fn1, fn2))
				fn_out = args.path_in + 'corrmatrix/%04d_%04d.txt'%(fn1,fn2)
				os.system('python %s %s %s -out %s' %(DM, im1, im2, fn_out)) 

	#Run DeepFlow
	#for i in range(nF):
	#	im1 = args.dir_out + 'refframes/frame_%04d.png'%args.ref_frames[i]
	#	for j in range(nF):
	#		if i != j:
	#			print("DeepFlow between frame %d and %d" %(args.ref_frames[i], args.ref_frames[j]))
	#			im2 = args.dir_out + 'refframes/frame_%04d.png'%args.ref_frames[j]
	#			fn_out = args.dir_out + 'corrmatrix/%04d_%04d.flo'%(args.ref_frames[i],args.ref_frames[j])
	#			matches = args.dir_out + 'corrmatrix/%04d_%04d.txt'%(args.ref_frames[i],args.ref_frames[j])
	#			os.system(DF + ' %s %s %s -match %s' %(im1, im2, fn_out, matches)) 

	#Run MFSF
	#for idx,ref in enumerate(args.ref_frames):
	#	if idx == 0:
	#		start = 1
	#	else:
	#		start = args.ref_frames[idx-1]
	#	if idx < nF-1:
	#		end = args.ref_frames[idx+1]
	#	else:
	#		end = nframes
	#	#Call MFSF matlab script...
	#	call = 'matlab -r "startup; run_mfsf_df(\'%s\', \'%s\', %d, %d, %d); exit;"' \
	#			%(args.path_in, args.name, start, ref, end-start+1)
	#	os.system(call)

if __name__ == "__main__":
	sys.exit(main())

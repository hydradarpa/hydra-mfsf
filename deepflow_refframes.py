#!/usr/bin/env python
import sys 
import argparse 
import os.path 
import os

import numpy as np 
import libtiff as lt 
from libtiff import TIFF

DM = '~/deepmatching/deep_matching_gpu.py'
DF = '~/deepflow/deepflow2-static'

def main():
	usage = """deepflow_refframes.py [input_tiff] [output_dir] [ref frame 1] [ref frame 2] <ref frame 3> ...

Normalize intensity of tiff stack 

Example: 
./deepflow_refframes.py "../hydra/video/20160412/stk_0001_Substack (1-5000).tif" ./register/20160412stk0001/ 1 40 80 120 

For help:
./deepflow_refframes.py -h 

Ben Lansdell
10/27/2016
"""

	parser = argparse.ArgumentParser()
	parser.add_argument('fn_in', help='input tiff file')
	parser.add_argument('dir_out', help='output directory')
	parser.add_argument('ref_frames', help='reference frames', nargs = '+', type = int)
	args = parser.parse_args()

	#Test code
	class Args:
		pass 
	args = Args()
	args.fn_in = '../hydra/video/20160412/stk_0001_Substack (1-5000).tif'
	args.ref_frames = [1, 40, 80, 120, 160]
	args.dir_out = './register/20160412stk0001/'
	args.name = '20160412stk0001'
	args.path_in = '../hydra/video/20160412/stk_0001/';

	#args.ref_frames = [1, 2, 3]
	#args.dir_out = './register/test/'

	nF = len(args.ref_frames)
	images = []

	#Make video directory
	if not os.path.isdir(args.dir_out + 'refframes'):
		os.makedirs(args.dir_out + 'refframes')
	if not os.path.isdir(args.dir_out + 'corrmatrix'):
		os.makedirs(args.dir_out + 'corrmatrix')
	if not os.path.isdir(args.dir_out + 'mfsf'):
		os.makedirs(args.dir_out + 'mfsf')
	#Use convert to extract frames

	# to open a tiff file for reading:
	tif = TIFF.open(args.fn_in, mode='r')
	info = tif.info()
	nframes = int(info.split('\n')[2].split('=')[1])

	for idx, image in enumerate(tif.iter_images()): # do stuff with image
		if idx in args.ref_frames:
			images.append(image)
	tif.close()

	for frame in args.ref_frames:
		os.system('convert "%s"[%d] -auto-level -depth 8 %s/refframes/frame_%04d.tif'%(args.fn_in, frame, args.dir_out, frame))
		os.system('convert "%s"[%d] -resize 50%% -auto-level -depth 8 %s/refframes/frame_%04d.png'%(args.fn_in, frame, args.dir_out, frame))

	#for frame, image in zip(args.ref_frames, images):
	#	fn_out = args.dir_out + 'refframes/frame_%04d.tif'%frame
	#	tif_out = TIFF.open(fn_out, mode='w')
	#	tif_out.write_image(image)
	#	tif_out.close()

	#Run DeepMatching
	for i in range(nF):
		im1 = args.dir_out + 'refframes/frame_%04d.png'%args.ref_frames[i]
		for j in range(nF):
			if i != j:
				print("DeepMatching between frame %d and %d" %(args.ref_frames[i], args.ref_frames[j]))
				im2 = args.dir_out + 'refframes/frame_%04d.png'%args.ref_frames[j]
				fn_out = args.dir_out + 'corrmatrix/%04d_%04d.txt'%(args.ref_frames[i],args.ref_frames[j])
				os.system('python %s %s %s -out %s' %(DM, im1, im2, fn_out)) 

	#Run DeepFlow
	for i in range(nF):
		im1 = args.dir_out + 'refframes/frame_%04d.png'%args.ref_frames[i]
		for j in range(nF):
			if i != j:
				print("DeepFlow between frame %d and %d" %(args.ref_frames[i], args.ref_frames[j]))
				im2 = args.dir_out + 'refframes/frame_%04d.png'%args.ref_frames[j]
				fn_out = args.dir_out + 'corrmatrix/%04d_%04d.flo'%(args.ref_frames[i],args.ref_frames[j])
				matches = args.dir_out + 'corrmatrix/%04d_%04d.txt'%(args.ref_frames[i],args.ref_frames[j])
				os.system(DF + ' %s %s %s -match %s' %(im1, im2, fn_out, matches)) 

	#Run MFSF
	for idx,ref in enumerate(args.ref_frames):
		if idx == 0:
			start = 1
		else:
			start = args.ref_frames[idx-1]
		if idx < nF-1:
			end = args.ref_frames[idx+1]
		else:
			end = nframes
		#Call MFSF matlab script...
		call = 'matlab -r "startup; run_mfsf_df(\'%s\', \'%s\', %d, %d, %d); exit;"' \
				%(args.path_in, args.name, start, ref, end-start+1)
		os.system(call)

if __name__ == "__main__":
	sys.exit(main())

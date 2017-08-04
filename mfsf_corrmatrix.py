#!/usr/bin/env python
import sys 
import argparse 
import os.path 
import os

import numpy as np 
from glob import glob 

from cvtools import readFlo

from scipy.io import loadmat 
import matplotlib.pyplot as plt
import cv2 

from lib.flow import flow_err_mfsf, flow_err_mfsf_rev

def main():
	usage = """mfsf_corrmatrix.py mfsf_dir name ...

Generate confidence maps based on optical flow estimation

Example: 
./mfsf_corrmatrix.py ./mfsf_output/ 20170219

For help:
./mfsf_corrmatrix.py -h 

Ben Lansdell
08/04/2017
"""

	parser = argparse.ArgumentParser()
	parser.add_argument('mfsf_in', help='output directory')
	parser.add_argument('res_dir', help='directory of results')
	parser.add_argument('name', help='project name')
	parser.add_argument('nframes', help='number of frames per block', type=int)
	args = parser.parse_args()

	#Test code for interactive dev
	#class Args:
	#	pass 
	#args = Args()
	#args.res_dir = './simmatrix/'
	#args.mfsf_in = './mfsf_output/'
	#args.name = '20170219'
	#args.nframes = 250



	threshold = 4

	#Get the corresponding list of refframes to compute error for...
	refframes = [int(i.split('_')[1].split('.')[0]) for i in glob(args.res_dir + '/' + args.name + '/refframes/*.png')]
	refframes.sort()
	nF = len(refframes)

	print refframes

	#Load MFSF results
	for idx in range(len(refframes)-1):
		r1 = refframes[idx]
		r2 = refframes[idx+1]
		print("Computing MFSF error between ref frames %d and %d" %(r1, r2))
		fn_in1 = args.mfsf_in + '/' + args.name + '/stack%04d_nref%d_nframe%d/result.mat'%(idx+1, 1, args.nframes)
		fn_in2 = args.mfsf_in + '/' + args.name + '/stack%04d_nref%d_nframe%d/result.mat'%(idx+1, args.nframes, args.nframes)
		
		fwderr = flow_err_mfsf(fn_in1, fn_in2)
		fwdtracked = fwderr < threshold
		reverr = flow_err_mfsf_rev(fn_in1, fn_in2)
		revtracked = reverr < threshold

		fn_out = args.res_dir + '/' + args.name + '/corrmatrix/%04d_%04d_mfsf_err.npz'%(r1, r2)
		np.savez(fn_out, fwderr = fwderr, fwdtracked = fwdtracked)
		fn_out = args.res_dir + '/' + args.name + '/corrmatrix/%04d_%04d_mfsf_err.npz'%(r2, r1)
		np.savez(fn_out, fwderr = reverr, fwdtracked = revtracked)

if __name__ == "__main__":
	sys.exit(main())
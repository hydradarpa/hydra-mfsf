#!/usr/bin/env python
import sys 
import argparse 
from lib.stitcher import Stitcher
from scipy.io import loadmat, savemat 
import numpy as np 
import os

from matplotlib import pyplot as plt

from vispy import gloo
from vispy import app

import gc 

def main():
	usage = """stitch_mfsf.py [output_matfile] [input_matfile 1] [input_matfile 2] <input_matfile 3> ...

Stitch together separate MFSF optic flow fields from separate videos into the one flow field 
whose coordinates are relative to the reference frame in the first video. This can be used for
tracking objects marked in the first video through later videos.

Example: 
./stitch_mfsf.py [stk_1-2.mat] [results_stk_0001.mat] [results_stk_0002.mat]

For help:
./stitch_mfsf.py -h 

Ben Lansdell
10/12/2016
"""

	parser = argparse.ArgumentParser()
	parser.add_argument('fn_out', help='output mat file')
	parser.add_argument('flow_in', help='input mat files from MFSF', nargs = '+')
	args = parser.parse_args()

	#Test code
	#class Args:
	#	pass 
	#args = Args()
	#args.fn_out = './stitched/stk_0001-0008/'
	#args.flow_in = ['./mfsf_output/stack0001_nref100_nframe250/result.mat',\
	#				'./mfsf_output/stack0002_nref100_nframe250/result.mat',\
	#				'./mfsf_output/stack0003_nref100_nframe250/result.mat',\
	#				'./mfsf_output/stack0004_nref100_nframe250/result.mat',\
	#				'./mfsf_output/stack0005_nref100_nframe250/result.mat',\
	#				'./mfsf_output/stack0006_nref100_nframe250/result.mat',\
	#				'./mfsf_output/stack0007_nref100_nframe250/result.mat',\
	#				'./mfsf_output/stack0008_nref100_nframe250/result.mat']

	if not os.path.exists(args.fn_out):
		os.makedirs(args.fn_out)

	nV = len(args.flow_in)
	if nV != 2:
		print("Specify more than 1 MFSF results file")
		#return 

	nref = np.zeros(nV)
	nF = np.zeros(nV)

	#Load first video and find last frame's coordinates
	a = loadmat(args.flow_in[0])
	params = a['parmsOF']
	u0 = a['u']
	v0 = a['v']
	info = a['info']
	nF[0] = u0.shape[2]	
	nx = u0.shape[0]
	ny = u0.shape[1]

	#Write this data to a mat file
	mdict = {'u':u0, 'v':v0, 'parmsOF':params, 'info':info}
	savemat(args.fn_out+'/mfsf_000.mat', mdict)

	#Load in optic flow data of the remaining videos
	for vidx in range(1,nV):
		#Load MFSF data
		a = loadmat(args.flow_in[vidx])	
		params = a['parmsOF']
		nF[vidx] = a['u'].shape[2]	

		u1 = a['u']
		v1 = a['v']
		#Make a Stitcher. Takes the second set's flow data as input
		thestitch = Stitcher(u1, v1)
		(u, v) = thestitch.run(u0, v0)

		#Could just update the present object instead....
		del thestitch
		unreachable = gc.collect()
		#self = thestitch

		u0 = u.copy()
		v0 = v.copy()

		#Save output matrix
		mdict = {'u':u, 'v':v, 'parmsOF':params, 'info':info}
		savemat(args.fn_out + '/mfsf_%03d.mat'%vidx, mdict)
		
if __name__ == "__main__":
	sys.exit(main())

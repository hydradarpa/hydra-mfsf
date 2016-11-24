#!/usr/bin/env pythonx
import sys 
import argparse 
from stitcher import Stitcher
from scipy.io import loadmat, savemat 
import numpy as np 

from matplotlib import pyplot as plt

from vispy import gloo
from vispy import app

#def main():
#	usage = """stitch_mfsf.py [output_matfile] [input_matfile 1] [input_matfile 2] <input_matfile 3> ...
#
#Stitch together separate MFSF optic flow fields from separate videos into the one flow field 
#whose coordinates are relative to the reference frame in the first video. This can be used for
#tracking objects marked in the first video through later videos.
#
#Example: 
#./stitch_mfsf.py [stk_1-2.mat] [results_stk_0001.mat] [results_stk_0002.mat]
#
#For help:
#./stitch_mfsf.py -h 
#
#Ben Lansdell
#10/12/2016
#"""
#
#	parser = argparse.ArgumentParser()
#	parser.add_argument('fn_out', help='output mat file')
#	parser.add_argument('flow_in', help='input mat files from MFSF', nargs = '+')
#	args = parser.parse_args()

	#Test code
	class Args:
		pass 
	args = Args()
	args.fn_out = './test_stitch.mat'
	args.flow_in = ['./mfsf_output/stack0001_nref100_nframe250/result.mat',\
					'./mfsf_output/stack0002_nref100_nframe250/result.mat']

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

	#Load remained of MFSF data to get total frames
	for vidx in range(1,nV):
		a = loadmat(args.flow_in[vidx])	
		nF[vidx] = a['u'].shape[2]	

	#Initialize data
	us = np.zeros((nx, ny, np.sum(nF)))
	vs = np.zeros((nx, ny, np.sum(nF)))
	us[:,:,0:nF[0]] = u0
	vs[:,:,0:nF[0]] = v0

	#Load in optic flow data
	for vidx in range(1,nV):
		#vidx = 1
		#Load MFSF data
		a = loadmat(args.flow_in[vidx])	
		params = a['parmsOF']
		u1 = a['u']
		v1 = a['v']
		#Make a Stitcher
		thestitch = Stitcher(u1, v1)

		self = thestitch
		(u, v) = thestitch.run(u0, v0)

		us[:,:,np.sum(nF[0:vidx]):np.sum(nF[0:vidx+1])] = u
		vs[:,:,np.sum(nF[0:vidx]):np.sum(nF[0:vidx+1])] = v

	#Save output matrix
	mdict = {'u':us, 'v':vs, 'parmsOF':params, 'info':info}
	savemat(args.fn_out, mdict)
		
if __name__ == "__main__":
	sys.exit(main())

#!/usr/bin/env python
import sys 
import argparse 
import os.path 
import os

import numpy as np 
from glob import glob 

from cvtools import readFlo

import matplotlib.pyplot as plt
import cv2 
from flow import flow_err_deepflow

def main():
	usage = """deepflow_corrmatrix.py [register_dir]...

Generate confidence maps based on optical flow estimation

Example: 
./deepflow_corrmatrix.py ./register/20160412stk0001/

For help:
./deepflow_corrmatrix.py -h 

Ben Lansdell
10/30/2016
"""

	parser = argparse.ArgumentParser()
	parser.add_argument('dir_in', help='output directory')
	args = parser.parse_args()

	#Test code for interactive dev
	#class Args:
	#	pass 
	#args = Args()
	#args.dir_in = './register/20160412stk0001/'
	#args.dir_in = './register/20160412stk0001-0008/'
	#args.dir_in = './register/test/'

	#Get the set of reference frames...
	refframes = [int(i[-8:-4]) for i in glob(args.dir_in + 'refframes/*.png')]

	#refframes = [1, 501, 1001, 1501]

	nF = len(refframes)

	threshold = 4
	radius = 6

	#Load DeepFlow results
	for r1 in refframes:
		for r2 in refframes:
			if r1 != r2:

				print("Comparing error in DeepFlow between frame %d and %d" %(r1, r2))
				fn_in1 = args.dir_in + 'corrmatrix/%04d_%04d.flo'%(r1, r2)
				fn_in2 = args.dir_in + 'corrmatrix/%04d_%04d.flo'%(r2, r1)

				fwderr = flow_err_deepflow(fn_in1, fn_in2)
				fwdtracked = fwderr < threshold

				#Save error matrices
				fn_out = args.dir_in + 'corrmatrix/%04d_%04d_deepflow_err.npz'%(r1, r2)
				np.savez(fn_out, fwderr = fwderr, fwdtracked = fwdtracked)

if __name__ == "__main__":
	sys.exit(main())
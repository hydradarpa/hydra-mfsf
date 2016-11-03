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
	class Args:
		pass 
	args = Args()
	args.dir_in = './register/20160412stk0001/'

	#args.dir_in = './register/test/'


	#Get the set of reference frames...
	refframes = [int(i[-8:-4]) for i in glob(args.dir_in + 'refframes/*.png')]
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

				flow1 = readFlo(fn_in1)
				flow2 = readFlo(fn_in2)

				nx = flow1.shape[0]
				ny = flow1.shape[1]

				#Flip x and y flow
				flow1 = np.transpose(flow1, [1,0,2])
				flow2 = np.transpose(flow2, [1,0,2])
				flow1 = flow1[:,:,::-1]
				flow2 = flow2[:,:,::-1]

				#Perform mapping and then reverse mapping, then perform reverse mapping then mapping
				#Make mesh grid
				fwdmeshy, fwdmeshx = [a.astype(np.float32) for a in np.meshgrid(np.arange(nx), np.arange(ny))]

				#Perturb mesh grid by forward flow 
				#Round to integers 
				fwdx = fwdmeshx + np.ceil(flow1[:,:,0])
				fwdy = fwdmeshy + np.ceil(flow1[:,:,1])
				fwdx = np.maximum(0, np.minimum(nx-1, fwdx))
				fwdy = np.maximum(0, np.minimum(nx-1, fwdy))
				#Look up flow field using this perturbed map
				fwdremapx = fwdx + flow2[fwdx.astype(int),fwdy.astype(int),0]
				fwdremapy = fwdy + flow2[fwdx.astype(int),fwdy.astype(int),1]
				fwdremapx -= fwdmeshx 
				fwdremapy -= fwdmeshy 
				fwderr = np.sqrt(fwdremapx**2 + fwdremapy**2)
				fwdtracked = fwderr < threshold

				#Save error matrices
				fn_out = args.dir_in + 'corrmatrix/%04d_%04d_deepflow_err.npz'%(r1, r2)
				np.savez(fn_out, fwderr = fwderr, fwdtracked = fwdtracked)

if __name__ == "__main__":
	sys.exit(main())
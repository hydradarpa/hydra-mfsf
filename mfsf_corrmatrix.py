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

def main():
	usage = """mfsf_corrmatrix.py [register_dir]...

Generate confidence maps based on optical flow estimation

Example: 
./mfsf_corrmatrix.py ./register/20160412stk0001/

For help:
./mfsf_corrmatrix.py -h 

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
	refframes = sorted(refframes)

	threshold = 4
	radius = 6

	#Load MFSF results
	for idx in range(len(refframes)-1):
		r1 = refframes[idx]
		r2 = refframes[idx+1]
		print("Computing MFSF error between ref frames %d and %d" %(r1, r2))
		fn_in1 = args.dir_in + 'mfsf/ref_%d/result.mat'%(r1)
		fn_in2 = args.dir_in + 'mfsf/ref_%d/result.mat'%(r2)

		#flow1 = readFlo(fn_in1)
		#flow2 = readFlo(fn_in2)
		
		#Load MFSF data
		a = loadmat(fn_in1)				
		params = a['parmsOF']
		u = a['u']
		v = a['v']
		nx = u.shape[0]
		ny = u.shape[1]
		flow1 = np.zeros((nx, ny, 2))
		flow1[:,:,0] = u[:,:,-1]
		flow1[:,:,1] = v[:,:,-1]

		a = loadmat(fn_in2)				
		params = a['parmsOF']
		u = a['u']
		v = a['v']
		flow2 = np.zeros((nx, ny, 2))
		flow2[:,:,0] = u[:,:,0]
		flow2[:,:,1] = v[:,:,0]

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

		revmeshy, revmeshx = [a.astype(np.float32) for a in np.meshgrid(np.arange(nx), np.arange(ny))]
		#Perturb mesh grid by forward flow 
		#Round to integers 
		revx = revmeshx + np.ceil(flow2[:,:,0])
		revy = revmeshy + np.ceil(flow2[:,:,1])
		revx = np.maximum(0, np.minimum(nx-1, revx))
		revy = np.maximum(0, np.minimum(nx-1, revy))
		#Look up flow field using this perturbed map
		revremapx = revx + flow1[revx.astype(int),revy.astype(int),0]
		revremapy = revy + flow1[revx.astype(int),revy.astype(int),1]
		revremapx -= revmeshx 
		revremapy -= revmeshy 
		reverr = np.sqrt(revremapx**2 + revremapy**2)
		revtracked = reverr < threshold

		#Save error matrices
		fn_out = args.dir_in + 'corrmatrix/%04d_%04d_mfsf_err.npz'%(r1, r2)
		np.savez(fn_out, fwderr = fwderr, fwdtracked = fwdtracked, reverr = reverr,\
			revtracked = revtracked)

if __name__ == "__main__":
	sys.exit(main())

#!/usr/bin/env pythonx
import sys 
import argparse 
from lib.stitcher import Stitcher, StitcherReverse
from scipy.io import loadmat, savemat 
import numpy as np 
import os
import h5py 
import gc 

from matplotlib import pyplot as plt

from vispy import gloo
from vispy import app

import cv2

from cvtools import readFlo

def read_MFSF(fn_in):
	#For both v1's MFSF rframe data run the stitcher
	try:
		a = loadmat(fn_in)	
		params = a['parmsOF']
		u1 = a['u']
		v1 = a['v']
	except:
		print "Failed to read using loadmat, using hdf5 library to read %s"%fn_in
		f = h5py.File(fn_in1,'r')
		#Note the x and y axes may need to be switched here...
		u1 = np.transpose(np.array(f.get('u')), (1,2,0))
		v1 = np.transpose(np.array(f.get('v')), (1,2,0))
	return (u1,v1)

def continuation(path_in, mfsf_in, iframes, rframes):
	usage = """Continue MFSF optic flow fields from separate videos into the one flow field 
whose coordinates are relative to a set of reference frames specified.

Example: 
./continue_dm_mfsf.py --rframes 1,501 --iframes 1,251,501,751 ./simmatrix/20160412/ ./mfsf_output/

For help:
./continue_dm_mfsf.py -h 

Ben Lansdell
1/14/2017
"""

	#Test code
	path_in = './simmatrix/20160412/seg_admm/gpu_MS_lambda_1.00e-04_rho_1.00e-03_niter_3000.npy'
	name = './simmatrix/20160412/'
	mfsf_in = './mfsf_output/'
	iframes = [1, 251, 501, 751, 1001, 1251, 1501]
	rframes = [1, 501] 

	dm_frames = [1]
	forward_mfsf = [[501, 751], [1251, 1501]]
	reverse_mfsf = [[501, 251], [1251, 1001]]

	#Prepare output directory 
	dr = name + './continuation_mfsf/'
	if not os.path.exists(dr):
	    os.makedirs(dr)

	#Load MS segmenting results for each reference frame
	u_s = np.load(path_in)

	nR = len(rframes)
	nF = len(iframes)
	nD = len(dm_frames)

	#For each DM file
	for vidx2 in range(nD):
		vidx = np.nonzero(np.array(iframes) == dm_frames[vidx2])[0][0]
		#Load MFSF data
		fn_in = mfsf_in + 'stack%04d_nref100_nframe250/result.mat'%(vidx+1)

		fn2 = iframes[vidx]

		print "Continuing video with DeepMatching (frame %d)"%fn2
		(u1, v1) = read_MFSF(fn_in)

		#Continue paths for both reference frames 
		#and save a mask of which paths should actually be continued based on 
		#segmentation 
		seg = np.argmax(cv2.resize(u_s[:,:,:,vidx], u1.shape[0:2]), axis = 2)

		for k in range(nR):
			fn1 = rframes[k]
			#Load in flow results for each reference frame to the iframe 
			if fn1 != fn2:
				#Make a Stitcher. Takes the second set's flow data as input
				thestitch = Stitcher(u1, v1)
				fn_in = name + '/corrmatrix/%04d_%04d.flo'%(fn1,fn2)
				flow = readFlo(fn_in)
				u0 = flow[:,:,0]
				v0 = flow[:,:,1]
				(u, v) = thestitch.run(u0, v0)	
				del thestitch
				unreachable = gc.collect()
				#Threshold and label paths that are to be continued with this rframe
				mask = (seg == k)
			else:
				u = u1 
				v = v1 
				mask = np.ones(u1.shape[0:2])

			#Save output matrix
			mdict = {'u':u, 'v':v, 'mask':mask}
			savemat(dr + '/mfsf_r_%04d_l_%04d_nref100.mat'%(fn1, fn2), mdict)
		
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('path_in', help='input directory with frames already placed in it')
	parser.add_argument('mfsf_in', help='input directory with mfsf output data already placed in it')
	parser.add_argument('--rframes', help='list of global reference frames. Provide as list of integers without space (e.g. 1,2,3,4)', type = str)
	parser.add_argument('--iframes', help='list of intermediate iframes. Provide as list of integers without space (e.g. 1,2,3,4)', type = str)
	args = parser.parse_args()

	iframes = [int(i) for i in args.iframes.split(',')]
	refframes = [int(i) for i in args.rframes.split(',')]

	continuation(args.path_in, args.mfsf_in, iframes, refframes)

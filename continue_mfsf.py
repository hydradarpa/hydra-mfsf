#!/usr/bin/env pythonx
import sys 
import argparse 
from lib.stitcher import Stitcher
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

def continuation(path_in, name, mfsf_in, iframes, rframes):
	usage = """Continue MFSF optic flow fields from separate videos into the one flow field 
whose coordinates are relative to a set of reference frames specified.

Example: 
./continue_mfsf.py --rframes 1,501 --iframes 1,251,501,751 ./simmatrix/20160412/ ./mfsf_output/

For help:
./continue_mfsf.py -h 

Ben Lansdell
1/5/2017
"""

	#Test code
	#path_in = './simmatrix/20160412/seg_admm/gpu_MS_lambda_1.00e-04_rho_1.00e-03_niter_3000.npy'
	#name = './simmatrix/20160412/'
	#mfsf_in = './mfsf_output/'
	#iframes = [1, 251, 501, 751, 1001, 1251, 1501, 1751, 2001, 2251, 2501, 2751, 3001, 3251, 3501, 3751, 4001,\
	#			4251, 4501, 4751]
	#rframes = [1, 501] 

	#Prepare output directory 
	
	dr = name + './continuation/'
	if not os.path.exists(dr):
	    os.makedirs(dr)

	#Load MS segmenting results for each reference frame
	u_s = np.load(path_in)

	nR = len(rframes)
	nF = len(iframes)

	#For each MFSF file
	for vidx in range(nF):
		#Load MFSF data
		fn_in = mfsf_in + 'stack%04d_nref1_nframe250/result.mat'%(vidx+1)

		fn2 = iframes[vidx]

		print "Continuing video (frame %d)"%fn2

		try:
			a = loadmat(fn_in)	
			params = a['parmsOF']
			u1 = a['u']
			v1 = a['v']
		except NotImplementedError:
			print "Failed to read using loadmat, using hdf5 library to read %s"%fn_in
			f = h5py.File(fn_in,'r')
			#Note the x and y axes may need to be switched here...
			u1 = np.transpose(np.array(f.get('u')), (1,2,0))
			v1 = np.transpose(np.array(f.get('v')), (1,2,0))

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
			savemat(dr + '/mfsf_r_%04d_l_%04d.mat'%(fn1, fn2), mdict)
		
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('path_in', help='input directory with frames already placed in it')
	parser.add_argument('name', help='name of video project')
	parser.add_argument('mfsf_in', help='input directory with mfsf output data already placed in it')
	parser.add_argument('--rframes', help='list of global reference frames. Provide as list of integers without space (e.g. 1,2,3,4)', type = str)
	parser.add_argument('--iframes', help='list of intermediate iframes. Provide as list of integers without space (e.g. 1,2,3,4)', type = str)
	args = parser.parse_args()

	iframes = [int(i) for i in args.iframes.split(',')]
	refframes = [int(i) for i in args.rframes.split(',')]

	continuation(args.path_in, args.name, args.mfsf_in, iframes, refframes)

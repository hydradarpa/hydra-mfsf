#!/usr/bin/env python
import sys 
import argparse 
from scipy.io import loadmat, savemat 
import numpy as np 
import os
import gc 
import cv2
import glob
from lib.warper import Warper

def main(args):
	usage = """Dewarp a set of points according to MFSF estimation

	Ben Lansdell
	5/25/2017
	"""
	#Test code:
	class Args:
		pass
	args = Args()
	#Format: ID, frame, x, y
	args.pts = './tracks/20160412/20160412_dupreannotation_stk0001.csv'
	args.output = './tracks/20160412/20160412_dupreannotation_stk0001_dewarp.csv'
	args.name = 'stack0001_nref100_nframe250'
	args.mfsf_dir = './mfsf_output'

	#Load points
	f_in = open(args.pts, 'r')
	pts = []
	for line in f_in:
		pt = [int(a) for a in line.split(',')]
		pts.append(pt)
	pts = np.array(pts)
	f_in.close()

	#Load MFSF data
	fn_in = '%s/%s/result.mat'%(args.mfsf_dir, args.name)
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

	nF = u1.shape[2]

	#Create Warper
	u = u1[:,:,0]
	v = v1[:,:,0]
	warper = Warper(u[:,:,np.newaxis], v[:,:,np.newaxis])
	frame = 0
	frame_pts = pts[pts[:,1]==frame,2:4]
	_ = warper.run(frame_pts)

	warped_pts = np.zeros((0,4))
	#For each frame, update warping object and dewarp points
	for frame in range(nF):
		print "Dewarping points in frame %d"%frame
		frame_pts = pts[pts[:,1]==frame,2:4]
		nC = frame_pts.shape[0]
		u = u1[:,:,frame]
		v = v1[:,:,frame]
		warper.update_flow(u[:,:,np.newaxis], v[:,:,np.newaxis])
		warped_pt = warper.run(frame_pts)
		warped_pts = np.vstack((warped_pts, np.hstack((np.arange(nC)[:,np.newaxis], frame*np.ones((nC, 1)), warped_pt))))

	#Save as new CSV
	numpy.savetxt(args.output, warped_pts, delimiter=",")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('name', help='name of MFSF data to use. Must be found in [mfsf_dir]/[name]/result.mat')
	parser.add_argument('pts', help='file with set of points to correct. CSV with format: frame,x,y')
	parser.add_argument('output', help='output file name')
	parser.add_argument('--mfsf_dir', help='', type = str, default='./mfsf_output/')
	args = parser.parse_args()
	main(args)
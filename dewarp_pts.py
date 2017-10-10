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

from matplotlib import pyplot, ion
from matplotlib.pyplot import pause
from time import sleep

def main(args):
	usage = """Dewarp a set of points according to MFSF estimation

	Ben Lansdell
	5/25/2017
	"""
	#Test code:
	ion()

	class Args:
		pass
	args = Args()
	#Format: ID, frame, x, y
	#args.pts = './tracks/20160412/20160412_dupreannotation_stk0001.csv'
	#args.output = './tracks/20160412/20160412_dupreannotation_stk0001_dewarp.csv'

	args.pts = './tracks/20160412/detections.csv'
	args.output = './tracks/20160412/detections_dewarp.csv'

	args.name = 'stack0001_nref100_nframe250'
	args.mfsf_dir = './mfsf_output'

	#Load points
	f_in = open(args.pts, 'r')
	title = '\n'
	pts = []
	for idx, line in enumerate(f_in):
		if idx > 0:
			pt = [int(a) for a in line.split(',')]
			pts.append(pt)
		else:
			title = line
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

	nF = min(u1.shape[2], max(pts[:,2]))

	#Create Warper
	u = u1[:,:,0]
	v = v1[:,:,0]
	warper = Warper(u[:,:,np.newaxis], v[:,:,np.newaxis])
	frame = 0
	frame_pts = pts[pts[:,2]==frame,5:7]
	_ = warper.run(frame_pts)

	warped_pts = np.zeros((0,4))
	#For each frame, update warping object and dewarp points
	for frame in range(nF):
		print "Dewarping points in frame %d"%frame
		frame_pts = pts[pts[:,2]==frame,5:7]
		frame_pts = np.fliplr(frame_pts)
		nC = frame_pts.shape[0]
		u = u1[:,:,frame]
		v = v1[:,:,frame]
		warper.update_flow(u[:,:,np.newaxis], v[:,:,np.newaxis])
		warped_pt = warper.run(frame_pts)
		pts[pts[:,2]==frame, 5:7] = warped_pt
		pts[pts[:,2]==frame, 0:2] = warped_pt/2

	#Save as new CSV
	np.savetxt(args.output, pts, delimiter=",", header = title, fmt = '%d')
	orig_pts = np.loadtxt(args.pts, delimiter=",", skiprows = 1)
	#pts = np.loadtxt(args.output, delimiter=",")

	#Plot the dewarped points to check they look right
	#Plot a bunch of frames
	f, (ax1, ax2) = pyplot.subplots(1, 2, sharey=True)
	
	for frm in range(nF):
		ax1.cla()
		ax2.cla()
		ax1.set_title('Frame %d. Transformed data'%frm)
		ax2.set_title('Original data')
		ax1.plot(pts[pts[:,2]==frm, 5], pts[pts[:,2]==frm, 6], '.')
		ax2.plot(orig_pts[orig_pts[:,2]==frm, 5], orig_pts[orig_pts[:,2]==frm, 6], '.')
		pyplot.draw()
		pause(0.5)
	#Looks good

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('name', help='name of MFSF data to use. Must be found in [mfsf_dir]/[name]/result.mat')
	parser.add_argument('pts', help='file with set of points to correct. CSV with format: frame,x,y')
	parser.add_argument('output', help='output file name')
	parser.add_argument('--mfsf_dir', help='', type = str, default='./mfsf_output/')
	args = parser.parse_args()
	main(args)
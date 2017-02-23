#!/usr/bin/env python
import sys, os
from lib.renderer import VideoStream, FlowStream, TIFFStream
from lib.kalman import IteratedMSKalmanFilter
from lib.distmesh_dyn import DistMesh
from scipy.io import loadmat 

import h5py 
import cv2 
import numpy as np 
import argparse
import gc 

def main():
	usage = """meshfmsf.py [vid_in] [flow_in] -threshold <> -gridsize <>

	Visualize results of MFSF run by meshing an object and 'tracking' it
		
	Ben Lansdell
	02/21/2017
	"""
	
	parser = argparse.ArgumentParser()
	parser.add_argument('vid_in', help='output mat file')
	parser.add_argument('flow_in', help='input mat files from MFSF')
	parser.add_argument('-threshold', help='intensities below this are not meshed', default=15)
	parser.add_argument('-gridsize', help='approximate gridsize for mesh', default=25)
	args = parser.parse_args()
	
	#fn_in='../hydra/video/20160412/stk_0001.avi'	
	#mfsf_in = './mfsf_output/stack0001_nref1_nframe250/'

	mfsf_in = args.flow_in
	fn_in = args.vid_in
	threshold = args.threshold 
	gridsize = args.gridsize

	dm_out = 'init_mesh.pkl'
	cuda = False

	imageoutput = mfsf_in + '/mesh/'
	#Make directory if needed...
	if not os.path.exists(imageoutput):
	    os.makedirs(imageoutput)
	
	#Load MFSF data
	try:
		a = loadmat(mfsf_in + '/result.mat')
		params = a['parmsOF']
		u = a['u']
		v = a['v']
		#Find reference frame 
		nref = params['nref'][0,0][0,0]
	except NotImplementedError:
		#Load using HDF package instead. This happens if the file is too big and had to be
		#saved using MATLAB's -v7.3 flag
		print "Failed to read using loadmat, using hdf5 library to read %s"%mfsf_in
		f = h5py.File(mfsf_in + '/result.mat','r')
		#Note the x and y axes may need to be switched here...
		u = np.transpose(np.array(f.get('u')), (1,2,0))
		v = np.transpose(np.array(f.get('v')), (1,2,0))
		params = f.get('parmsOF')
		#Figure out how to get the reference frame from the structure...
		nref = int(params['nref'][0][0])
	print "Loaded MFSF data"

	#Skip to this frame and create mesh 
	capture = TIFFStream(fn_in, threshold)
	
	nx = capture.nx
	nF = capture.nframes
	
	for idx in range(nF):
		print 'Loading frame', idx
		ret, frame, mask = capture.read()
		if idx == nref:
			refframe = frame.copy()
			masks = mask
	
	distmesh = DistMesh(refframe, h0 = gridsize)
	if not os.path.exists(mfsf_in + dm_out):
		distmesh.createMesh(ctrs, fd, refframe, plot = True)
		#Save this distmesh and reload it for quicker testing
		distmesh.save(mfsf_in + dm_out)
	else:
		distmesh.load(mfsf_in + dm_out)
	
	refpositions = distmesh.p
	
	#Create dummy input for flow frame
	flowframe = np.zeros((nx, nx, 2))
	
	#Create Kalman Filter object to store mesh and make use of plotting functions
	kf = IteratedMSKalmanFilter(distmesh, refframe, flowframe, cuda = cuda)
	
	#Perturb mesh points according to MFSF flow field and save screenshot output
	nF = min(u.shape[2], nF)
	N = kf.size()/4

	del capture
	gc.collect()
	capture2 = TIFFStream(fn_in, threshold)

	
	for idx in range(nF):
		#Update positions based on reference positions and flow field
		print("Visualizing frame %d" % idx)
		ret, frame, mask = capture2.read() 
		y_im = frame.astype(np.dtype('uint8'))
		y_m = mask.astype(np.dtype('uint8'))
		kf.state.renderer.update_frame(y_im, flowframe, y_m)
		dx = u[refpositions[:,1].astype(int), refpositions[:,0].astype(int), idx]
		dy = v[refpositions[:,1].astype(int), refpositions[:,0].astype(int), idx]
		X = refpositions.copy()
		X[:,0] += dx
		X[:,1] += dy
		kf.state.X[0:2*N] = np.reshape(X, (-1,1))
		kf.state.refresh()
		kf.state.render()
		kf.state.renderer.screenshot(saveall=True, basename = imageoutput + '_frame_%03d' % idx)
		
	#Make a video
	print 'Making movie'
	overlayoutput = mfsf_in + '/mesh_overlay/'
	if not os.path.exists(overlayoutput):
	    os.makedirs(overlayoutput)
	
	for idx in range(nF):
		os.system('cp ' + imageoutput + '_frame_%03d_overlay* '%idx + overlayoutput+'frame_%03d.png'%idx)
	
	avconv = 'avconv -framerate 5 -i ' + overlayoutput + 'frame_%03d.png -c:v mpeg4 -qscale 8 -y'
	os.system(avconv + ' ' + overlayoutput + 'output.avi')

if __name__ == "__main__":
	sys.exit(main())
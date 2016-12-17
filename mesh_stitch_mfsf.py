#!/usr/bin/env python
import sys, os
from renderer import VideoStream, TIFFStream
from kalman import IteratedMSKalmanFilter
from distmesh_dyn import DistMesh
from scipy.io import loadmat 

from imgproc import findObjectThreshold
from matplotlib import pyplot as plt

import cv2 
import numpy as np 
from glob import glob 

name='stk_0001-0008'
mfsf_in = './stitched/' + name + '/'
vid_path_in = '../hydra/video/20160412/combined/'
fn_ins = ['stk_0001.tif', 'stk_0002.tif', 'stk_0003.tif', 'stk_0004.tif', 'stk_0005.tif',\
			'stk_0006.tif', 'stk_0007.tif', 'stk_0008.tif']

mask_in = './stitched/frame_100_roi_body.png'
threshold = 2
cuda = True
gridsize = 25

dm_out = './stitched/darpa_mesh.pkl'

imageoutput = './stitched/' + name + '/mesh/'
#Make directory if needed...
if not os.path.exists(imageoutput):
    os.makedirs(imageoutput)

mfsf_matfiles = sorted(glob(mfsf_in + '*.mat'))
nV = len(mfsf_matfiles)

#Load MFSF data
a = loadmat(mfsf_matfiles[0])
params = a['parmsOF']

#Find reference frame 
nref = params['nref'][0,0][0,0]

tracking_mask = cv2.imread(mask_in)
tracking_mask = cv2.cvtColor(tracking_mask,cv2.COLOR_BGR2GRAY)

#Skip to this frame and create mesh 
capture = TIFFStream(vid_path_in + fn_ins[0], threshold)

nx = capture.nx
nF = capture.nframes

vid = np.zeros((nx, nx, nF))
masks = np.zeros((nx, nx, nF))
for idx in range(nF):
	print 'Loading frame', idx
	ret, frame, mask = capture.read()
	if idx == nref:
		refframe = frame.copy()
	vid[:,:,idx] = frame 
	masks[:,:,idx] = mask

#Generate ctrs and fd
(mask, ctrs, fd) = findObjectThreshold(tracking_mask, threshold = threshold)

#distmesh = DistMesh(refframe, h0 = gridsize)
#distmesh.createMesh(ctrs, fd, refframe, plot = True)
#Save this distmesh and reload it for quicker testing
#distmesh.save(dm_out)

distmesh = DistMesh(refframe, h0 = gridsize)
distmesh.load(dm_out)

refpositions = distmesh.p
#Create dummy input for flow frame
flowframe = np.zeros((nx, nx, 2))

#Create Kalman Filter object to store mesh and make use of plotting functions
kf = IteratedMSKalmanFilter(distmesh, refframe, flowframe, cuda = cuda)
N = kf.size()/4
totalframes = 0

for vidx in range(nV):

	#Load MFSF data
	a = loadmat(mfsf_matfiles[vidx])
	params = a['parmsOF']
	u = a['u']
	v = a['v']
	if vidx > 0:
		#Skip to this frame and create mesh 
		capture = TIFFStream(vid_path_in + fn_ins[vidx], threshold)
		nx = capture.nx
		nF = capture.nframes
		vid = np.zeros((nx, nx, nF))
		masks = np.zeros((nx, nx, nF))
		for idx in range(nF):
			print 'Loading frame', idx
			ret, frame, mask = capture.read()
			vid[:,:,idx] = frame 
			masks[:,:,idx] = mask
		#Perturb mesh points according to MFSF flow field and save screenshot output
		nF = u.shape[2]

	for idx in range(nF):
		#capture.seek(idx)
		#Update positions based on reference positions and flow field
		print("Visualizing frame %d" % idx)
		#ret, frame, y_im, y_m = capture2.read()
		y_im = vid[:,:,idx].astype(np.dtype('uint8'))
		y_m = masks[:,:,idx].astype(np.dtype('uint8'))
		#if not ret:
		#	print("End of stream encountered")
		#	break 
		kf.state.renderer.update_frame(y_im, flowframe, y_m)
		dx = u[refpositions[:,1].astype(int), refpositions[:,0].astype(int), idx]
		dy = v[refpositions[:,1].astype(int), refpositions[:,0].astype(int), idx]
		X = refpositions.copy()
		X[:,0] += dx
		X[:,1] += dy
		kf.state.X[0:2*N] = np.reshape(X, (-1,1))
		kf.state.refresh()
		kf.state.render()
		#kf.state.renderer.screenshot(saveall=True, basename = imageoutput + 'frame_%04d'%(totalframes+idx))
		wireframe = kf.state.renderer.wireframe()
		cv2.imwrite(imageoutput + 'frame_%04d.png'%(totalframes+idx), wireframe)

	totalframes += nF


#Make a video
print 'Making movie'
overlayoutput = './stitched/' + name + '/mesh_overlay/'
if not os.path.exists(overlayoutput):
    os.makedirs(overlayoutput)

os.system('cp ' + imageoutput + '_frame_*overlay* ' + overlayoutput)

avconv = 'avconv -i ' + overlayoutput + '_frame_%03d_overlay.png -c:v huffyuv -y'
os.system(avconv + ' ' + overlayoutput + 'output.avi')
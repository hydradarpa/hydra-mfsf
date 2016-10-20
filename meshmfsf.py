#!/usr/bin/env python
import sys, os
from renderer import VideoStream, FlowStream
from kalman import IteratedMSKalmanFilter
from distmesh_dyn import DistMesh
from scipy.io import loadmat 

import cv2 
import numpy as np 

fn_in='../hydra/video/johntest_brightcontrast_short_jpg.avi'
name='johntest_brightcontrast_short'
threshold = 15
cuda = True
gridsize = 25

#mfsf_in = './mfsf_output/stk_0001_mfsf_nref100/'
mfsf_in = './mfsf_output/johntest_flag_STD_0_alpha_20_nref100/'
dm_out = 'init_mesh.pkl'

imageoutput = mfsf_in + '/mesh/'
#Make directory if needed...
if not os.path.exists(imageoutput):
    os.makedirs(imageoutput)

#Load MFSF data
a = loadmat(mfsf_in + '/result.mat')

params = a['parmsOF']
u = a['u']
v = a['v']

#Find reference frame 
nref = params['nref'][0,0][0,0]

#Skip to this frame and create mesh 
capture = VideoStream(fn_in, threshold)

nx = int(capture.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
nF = int(capture.cap.get(cv2.CAP_PROP_FRAME_COUNT))

vid = np.zeros((nx, nx, nF))
masks = np.zeros((nx, nx, nF))
for idx in range(nF):
	print 'Loading frame', idx
	ret, frame, grayframe, mask = capture.read()
	if idx == nref:
		mask, ctrs, fd = capture.backsub()
		refframe = frame 
	vid[:,:,idx] = grayframe 
	masks[:,:,idx] = mask

#distmesh = DistMesh(refframe, h0 = gridsize)
#distmesh.createMesh(ctrs, fd, refframe, plot = True)
#Save this distmesh and reload it for quicker testing
#distmesh.save(mfsf_in + dm_out)

distmesh = DistMesh(refframe, h0 = gridsize)
distmesh.load(mfsf_in + dm_out)

refpositions = distmesh.p

#Create dummy input for flow frame
flowframe = np.zeros((nx, nx, 2))

#Create Kalman Filter object to store mesh and make use of plotting functions
kf = IteratedMSKalmanFilter(distmesh, refframe, flowframe, cuda = cuda)

#Perturb mesh points according to MFSF flow field and save screenshot output
nF = u.shape[2]
N = kf.size()/4

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
	kf.state.renderer.screenshot(saveall=True, basename = imageoutput + '_frame_%03d' % idx)


#Make a video
print 'Making movie'
overlayoutput = mfsf_in + '/mesh_overlay/'
if not os.path.exists(overlayoutput):
    os.makedirs(overlayoutput)

for idx in range(nF):
	os.system('cp ' + imageoutput + '_frame_%03d_overlay* '%idx + overlayoutput+'frame_%03d.png'%idx)

avconv = 'avconv -framerate 5 -i ' + overlayoutput + 'frame_%03d.png -c:v huffyuv -y'
os.system(avconv + ' ' + overlayoutput + 'output.avi')

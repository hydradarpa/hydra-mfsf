#!/usr/bin/env python
import sys, os
from renderer import VideoStream, FlowStream
from kalman import IteratedMSKalmanFilter
from distmesh_dyn import DistMesh
from scipy.io import loadmat 
from imgproc import findObjectThreshold, load_ground_truth
from matplotlib import pyplot as plt
import matplotlib

import cv2 
import numpy as np 
from numpy.random import choice, randint

font = {'family' : 'normal',
        'size'   : 28}

matplotlib.rc('font', **font)

fn_in = '../hydra/video/20160412/stk_0001_0002.avi'
mask_in = './stitched/frame_100_roi_body.png'
name='stack0001_darpa'
mfsf_in = './mfsf_output/stk_0001_mfsf_nref100/'
groundtruth_in = './analysis/stack_0001-1-620-groundtruthneurontracks.csv'

dm_out = 'darpa_mesh.pkl'

threshold = 2
cuda = True
gridsize = 40
nR = 10 		#Number of neurons to display per class
radius = 6
trackedthresh = 10
err_nx = 1136.
err_ny = 386.
dpi = 96.

imageoutput = mfsf_in + '/mesh_neurons/'
#Make directory if needed...
if not os.path.exists(imageoutput):
    os.makedirs(imageoutput)

#Load ground truth tracking structure for flow tracking
true_cells = load_ground_truth(groundtruth_in)
nC = len(true_cells[0][0].keys())

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
		refframe = frame 
	vid[:,:,idx] = grayframe 
	masks[:,:,idx] = mask

tracking_mask = cv2.imread(mask_in)
tracking_mask = cv2.cvtColor(tracking_mask,cv2.COLOR_BGR2GRAY)

(mask, ctrs, fd) = findObjectThreshold(tracking_mask, threshold = threshold)

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

#Load estimated neuron tracks
truepositions = np.zeros((nF, nC, 2))
estpositions = np.zeros((nF, nC, 2))
refpositions_cells = np.zeros((nC, 2))
distance_error = np.zeros((nF, nC))
max_error = np.zeros(nC)

for cell, loc in true_cells[0][nref].iteritems():
	refpositions_cells[cell-1,:] = loc 

for frame in range(nF):
	for cell, loc in true_cells[0][frame].iteritems():
		truepositions[frame,cell-1,:] = loc 

for idx in range(nF):
	#Update tracked positions
	dx = u[refpositions_cells[:,1].astype(int), refpositions_cells[:,0].astype(int), idx]
	dy = v[refpositions_cells[:,1].astype(int), refpositions_cells[:,0].astype(int), idx]
	estpositions[idx,:,:] = refpositions_cells.copy()
	estpositions[idx,:,0] += dx
	estpositions[idx,:,1] += dy
	for c in range(nC):
		tp = truepositions[idx,c,:]
		ep = estpositions[idx,c,:]
		distance_error[idx,c] = np.sqrt(np.sum((tp-ep)*(tp-ep)))
max_error = distance_error.max(0)

#Remove cells outside mask
inmask = mask[refpositions_cells[:,1].astype(int), refpositions_cells[:,0].astype(int)].astype(bool)
estpositions = estpositions[:,inmask,:]
distance_error = distance_error[:,inmask]
max_error = max_error[inmask]

nC = estpositions.shape[1]

#Select a random subset of tracked and untracked neurons to display
tracked = np.where(max_error < trackedthresh)[0]
untracked = np.where(max_error > trackedthresh)[0]

trackedchoice = choice(tracked, nR)
untrackedchoice = choice(untracked, nR)
tracked_estpositions = estpositions[:,trackedchoice,:]
tracked_distance_error = distance_error[:,trackedchoice]
untracked_estpositions = estpositions[:,untrackedchoice,:]
untracked_distance_error = distance_error[:,untrackedchoice]
nC = nR

#Pick some random colors
trackedcolor = np.array([0, 200, 50, 255])
errcolor = np.array([50, 50, 255, 255])
errcolor_rgb = np.array([255, 50, 50, 255])

fig1 = plt.figure(figsize = (1.5*err_nx/dpi, 1.5*err_ny/dpi), dpi = dpi)
ax1 = fig1.add_subplot(111)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='on') # labels along the bottom edge are off

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='minor',      # both major and minor ticks are affected
    labelbottom='off') # labels along the bottom edge are off

plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='minor',      # both major and minor ticks are affected
    labelleft='off') # labels along the bottom edge are off

plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left='off',      # ticks along the bottom edge are off
    right='off',         # ticks along the top edge are off
    labelleft='on') # labels along the bottom edge are off

for idx in range(nF):
	print("Visualizing frame %d" % idx)
	y_im = vid[:,:,idx].astype(np.dtype('uint8'))
	y_m = masks[:,:,idx].astype(np.dtype('uint8'))
	kf.state.renderer.update_frame(y_im, flowframe, y_m)
	dx = u[refpositions[:,1].astype(int), refpositions[:,0].astype(int), idx]
	dy = v[refpositions[:,1].astype(int), refpositions[:,0].astype(int), idx]
	X = refpositions.copy()
	X[:,0] += dx
	X[:,1] += dy
	kf.state.X[0:2*N] = np.reshape(X, (-1,1))
	kf.state.refresh()
	kf.state.render()
	wireframe = kf.state.renderer.wireframe()

	trackedwire = wireframe.copy()
	untrackedwire = wireframe.copy()

	#Display the neurons on top of the wireframe...
	for c in range(nC):
		center = tuple(tracked_estpositions[idx,c,:].astype(int))
		untrackedcenter = tuple(untracked_estpositions[idx,c,:].astype(int))
		#color = (colors[idx,c,:]*255).astype(np.uint8)
		trackedwire = cv2.circle(trackedwire, center, radius, (tuple(trackedcolor)), -1)
		untrackedwire = cv2.circle(untrackedwire, untrackedcenter, radius, (tuple(errcolor)), -1)
		untrackedwire = cv2.circle(untrackedwire, center, radius, (tuple(trackedcolor)), -1)

	#Save image
	fn_out = "%s/frame_%04d_untracked.png"%(imageoutput,idx)
	cv2.imwrite(fn_out, untrackedwire)
	fn_out = "%s/frame_%04d_tracked.png"%(imageoutput,idx)
	cv2.imwrite(fn_out, trackedwire)

	#Now also make the errors plot...
	#Tracked only
	fn_out = "%s/frame_tracked_errors_%04d.png"%(imageoutput,idx)
	ax1.cla()
	xpast = np.arange(idx)
	xfuture = range(idx, nF);
	ax1.plot(xpast/10., tracked_distance_error[xpast,:], linewidth = 3, color = tuple(trackedcolor/255.), zorder = 1)
	#ax1.plot(xfuture, distance_error[xfuture,:], linewidth = 0.5, color = (0.8, 0.8, 0.8))
	ax1.scatter(x = (idx-1)*np.ones(nC)/10., y = tracked_distance_error[max(0,idx-1),:], color = (0,0,0), s = 20, zorder = 2)
	ax1.set_ylim([0, 30])
	ax1.set_xlim([0, nF/10.])
	ax1.set_ylabel('tracking error (pixels)')
	ax1.set_xlabel('time (s)')
	#plt.show()
	ax1.set_xticks(np.array([10, 20]))
	ax1.set_yticks(np.array([0, 15, 30]))
	plt.savefig(fn_out, bbox_inches = 'tight')

	#Tracked and untracked
	fn_out = "%s/frame_untracked_errors_%04d.png"%(imageoutput,idx)
	ax1.cla()
	xpast = np.arange(idx)
	xfuture = range(idx, nF);
	ax1.plot(xpast/10., untracked_distance_error[xpast,:], linewidth = 3, color = tuple(errcolor_rgb/255.), zorder = 2)
	ax1.plot(np.arange(nF)/10., tracked_distance_error[:,:], linewidth = 3, color = tuple(trackedcolor/255.), zorder = 1)
	#ax1.plot(xfuture, distance_error[xfuture,:], linewidth = 0.5, color = (0.8, 0.8, 0.8))
	ax1.scatter(x = (idx-1)*np.ones(nC)/10., y = untracked_distance_error[max(0,idx-1),:], color = (0,0,0), s = 20, zorder = 3)
	ax1.set_ylim([0, 30])
	ax1.set_xlim([0, nF/10.])
	ax1.set_ylabel('tracking error (pixels)')
	ax1.set_xlabel('time (s)')
	ax1.set_xticks(np.array([10, 20]))
	ax1.set_yticks(np.array([0, 15, 30]))
	plt.savefig(fn_out, bbox_inches = 'tight')

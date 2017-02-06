#!/usr/bin/env python
import sys, os
from lib.renderer import VideoStream, TIFFStream
from lib.kalman import IteratedMSKalmanFilter
from lib.distmesh_dyn import DistMesh
from scipy.io import loadmat 

from lib.imgproc import findObjectThreshold
from matplotlib import pyplot as plt

import cv2 
import numpy as np 
from glob import glob 

from lib.imgproc import drawFaces

def continuation(path_in, mfsf_in, iframes, rframes):
	usage = """Continue MFSF optic flow fields from separate videos into the one flow field 
	whose coordinates are relative to a set of reference frames specified. Visualize results 
	by generating meshes of different colors -- to correspond to different reference frames.
	
	Example: 
	./continue_mesh.py --rframes 1,501 --iframes 1,251,501,751 ./simmatrix/20160412/ ./mfsf_output/ 
	
	For help:
	./continue_mesh.py -h 
	
	Ben Lansdell
	1/6/2017
	"""

	#Test code 
	vid_path_in = '../hydra/video/20160412/combined/'
	name = './simmatrix/20160412/'
	mfsf_in = name + '/continuation_mfsf/'
	#seg_in = name + '/seg_admm/gpu_MS_lambda_1.00e-04_rho_1.00e-03_niter_3000.npy'
	rframes = [1, 501] 
	iframes = [1]
	colors = np.array([[0, 255, 0], [255, 0, 0]])

	dm_frames = [1, 501, 1251]
	forward_mfsf = [[501, 751], [1251, 1501]]
	reverse_mfsf = [[501, 251], [1251, 1001]]

	threshold = 2
	cuda = True
	gridsize = 50

	mesh_out = name + '/mesh/'
	#Make directory if needed...
	if not os.path.exists(mesh_out):
	    os.makedirs(mesh_out)

	imageoutput = name + '/mesh_frames/'
	#Make directory if needed...
	if not os.path.exists(imageoutput):
	    os.makedirs(imageoutput)    

	masks = []
	refpositions = []
	faces = []

	#Load in mask images 
	for idx, fn in enumerate(rframes):
		dm_out = mesh_out + '/frame_%04d.pkl'%fn	
		mask_in = name + '/masks/mask_%04d.png'%fn

		masks.append(cv2.cvtColor(cv2.imread(mask_in), cv2.COLOR_BGR2GRAY))

		#Make the meshes
		#Generate ctrs and fd
		(m, ctrs, fd) = findObjectThreshold(masks[idx], threshold = threshold)

		d = DistMesh(m, h0 = gridsize)
		if not os.path.exists(dm_out):
			d.createMesh(ctrs, fd, m, plot = True)
			#Save this distmesh and reload it for quicker testing
			d.save(dm_out)
		else:
			d.load(dm_out)	

		refpositions.append(d.p)
		faces.append(d.t)

	refframe_files = sorted(glob(name + '/refframes/*.png'))
	mfsf_matfiles = sorted(glob(mfsf_in + '*.mat'))
	nV = len(refframe_files)

	#For each MFSF file, we will make a video coloring the mesh by 		
	for l,fn2 in enumerate(iframes):

		continued = []
		u = []
		v = []
		#Flow data ./simmatrix/20160412/continuation/mfsf_r_0001_l_4251.mat
		for r,fn1 in enumerate(rframes):
			fn = mfsf_in + '/mfsf_r_%04d_l_%04d_nref100.mat'%(fn1,fn2)
			a = loadmat(fn)
			c = a['mask']
			if len(c.shape) == 3:
				continued.append(a['mask'][:,:,0])
			else:
				continued.append(a['mask'])
			u.append(a['u'])
			v.append(a['v'])

		#Load video stream
		capture = TIFFStream(vid_path_in + 'stk_%04d.tif'%(l+1), threshold)
		nx = capture.nx
		nF = capture.nframes
	
		#Perturb initial meshes by their flow fields
		positions = []
		active_pts = []
		active_faces = []
		for r,fn1 in enumerate(rframes):
			dx = u[r][refpositions[r][:,1].astype(int), refpositions[r][:,0].astype(int), 0]
			dy = v[r][refpositions[r][:,1].astype(int), refpositions[r][:,0].astype(int), 0]
			X = refpositions[r].copy()
			X[:,0] += dx
			X[:,1] += dy
			positions.append(X)
			#Intersect perturbed points by labeled continuation data to get
			#the faces that we're going to display in this video
			act_pts = continued[r][X[:,1].astype(int), X[:,0].astype(int)]
			act_fcs = np.array([act_pts[f].all() for f in faces[r]])
			active_pts.append(act_pts)
			active_faces.append(act_fcs)

		#Then, for each frame, perturb the vertices by their respective flow fields
		#and draw the active faces from each reference frame a different color
		for idx in range(nF):
			print("Visualizing frame %d" % idx)
			ret, frame, _ = capture.read(backsub = False)
			frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

			for r,fn1 in enumerate(rframes):
				#Perturb the vertices according to the flow
				dx = u[r][refpositions[r][:,1].astype(int), refpositions[r][:,0].astype(int), idx]
				dy = v[r][refpositions[r][:,1].astype(int), refpositions[r][:,0].astype(int), idx]
				X = refpositions[r].copy()
				X[:,0] += dx
				X[:,1] += dy

				#Draw the active faces
				col = colors[r,:]
				drawFaces(frame, X, faces[r][active_faces[r]], col)
		
			#Save
			cv2.imwrite(imageoutput + 'l_%04d_frame_%04d.png'%(fn2,idx), frame)
	
		#Make a video
		print 'Making movie'
		overlayoutput = name + '/mesh_movies/'
		if not os.path.exists(overlayoutput):
		    os.makedirs(overlayoutput)
				
		avconv = 'avconv -i ' + imageoutput + 'l_' + '%04d'%fn2 + '_frame_%04d.png -c:v mpeg4 -qscale 7 -y'
		os.system(avconv + ' ' + overlayoutput + 'output_l_%04d.avi'%fn2)

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

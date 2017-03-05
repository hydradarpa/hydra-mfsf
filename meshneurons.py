#!/usr/bin/env python
import sys, os
from lib.renderer import VideoStream, TIFFStream
from lib.distmesh_dyn import Delaunay, orientation
from scipy.io import loadmat 

import distmesh.mlcompat as ml
import numpy.matlib as npml
import h5py 
import cv2 
import numpy as np 
import argparse
import gc 
import cv2
from lib.imgproc import drawGrid, findObjectThreshold
from lib.tracks import load_tracks_csv

def main():
	usage = """meshfneurons.py [vid_in] [tracks_csv] [name]

	Make animation of mesh with edges being the tracked neuron positions.
	To highlight the discontinuity in motion -- if it exists. 
		
	Ben Lansdell
	03/02/2017
	"""
	
	parser = argparse.ArgumentParser()
	parser.add_argument('vid_in', help='video file for animation')
	parser.add_argument('tracks_in', help='csv file with tracks')
	parser.add_argument('name', help='name to save video files')
	args = parser.parse_args()
	
	#Test code
	#fn_in='../hydra/video/20170202/20170202_8bit.tif'	
	#mfsf_in = './mfsf_output/20170202_16bit/'
	#gridsize = 50
	#threshold = 15

	threshold = 22
	name = args.name
	if name[-1] != '/':
		name += '/'
	tracks_in = args.tracks_in
	fn_in = args.vid_in

	#Skip to this frame and create mesh 
	capture = TIFFStream(fn_in, threshold)
	nx = capture.nx
	nF = capture.nframes

	#Load tracks
	ret, frame, mask = capture.read()
	(mask, ctrs, fd) = findObjectThreshold(mask, threshold = .5)

	tracks = load_tracks_csv(tracks_in)
	nC = len(tracks)
	p = np.zeros((nC, 2))
	for c in tracks.keys():
		p[c,:] = tracks[c][0][0:2]
	bars, t = Delaunay(p, fd)
	original_ori = orientation(p, t)
	nB = len(bars)

	imageoutput = name + 'mesh_neurons/'
	#Make directory if needed...
	if not os.path.exists(imageoutput):
	    os.makedirs(imageoutput)
	
	capture = TIFFStream(fn_in, threshold)
	
	for idx in range(nF):

		#For each edge in each 
		#Update positions based on reference positions and flow field
		print("Visualizing frame %d" % idx)
		ret, frame, mask = capture.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
		for c in tracks.keys():
			p[c,:] = tracks[c][idx][0:2]
		current_ori = orientation(p, t)

		invalid_ori = (current_ori*original_ori == -1)
		threeori = np.hstack((invalid_ori, invalid_ori, invalid_ori))

		bars = np.vstack((t[:, [0,1]],
							t[:, [1,2]],
							t[:, [2,0]]))          # Interior bars duplicated

		#Bars with bad orientation...
		invalid_bars = bars[threeori,:]
		invalid_bars.sort(axis=1)
		invalid_bars = ml.unique_rows(invalid_bars)

		bars.sort(axis=1)
		bars = ml.unique_rows(bars)              # Bars as node pairs

		colors = npml.repmat([0, 255, 0], nB, 1)
		for i,b in enumerate(bars):
			if b in invalid_bars:
				colors[i] = [0, 0, 255]

		drawGrid(frame, p, bars, cols = colors)
		cv2.imwrite(imageoutput + 'frame_%03d.png'%idx, frame)
		
	#Make a video
	print 'Making movie'
	overlayoutput = name + 'mesh_overlay/'
	if not os.path.exists(overlayoutput):
	    os.makedirs(overlayoutput)
	
	#avconv = 'avconv -i ' + imageoutput + 'frame_%03d.png -c:v mpeg4 -qscale 8 -y'
	avconv = 'avconv -i ' + imageoutput + 'frame_%03d.png -c:v mpeg4 -qscale 15 -y'
	os.system(avconv + ' ' + overlayoutput + 'meshneurons.mp4')

if __name__ == "__main__":
	sys.exit(main())
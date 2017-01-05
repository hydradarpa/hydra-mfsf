#!/usr/bin/env python
import sys 
import argparse 
import os.path 
import os

import numpy as np 
import libtiff as lt 
from libtiff import TIFF

from glob import glob 

from cvtools import readFlo

import cv2

import matplotlib.pyplot as plt 
import scipy
import pylab
import scipy.cluster.hierarchy as sch

def main():
	usage = """deepflow_simmatrix.py [input_dir]

Expects refframes directory in [input_dir] to be already filled with i-frames

Example: 
./deepflow_simmatrix_viz.py ./simmatrix/20160412/

For help:
./deepflow_simmatrix_viz.py -h 

Ben Lansdell
01/04/2016
"""

	parser = argparse.ArgumentParser()
	parser.add_argument('path_in', help='input directory with frames already placed in it')
	args = parser.parse_args()

	#Test code
	class Args:
		pass 
	args = Args()
	args.path_in = './simmatrix/20160412/'

	#Get all files in 
	iframes = sorted(glob(args.path_in + 'refframes/*.tif'))
	nF = len(iframes)

	nx_tile = 64

	#Load images
	images = []
	for fn_in in iframes: # do stuff with image
		# to open a tiff file for reading:
		tif = TIFF.open(fn_in, mode='r')
		image = tif.read_image()
		image = cv2.resize(image, (nx_tile, nx_tile))
		images.append(image)
		tif.close()

	D = np.zeros((nF,nF))

	#Run DeepFlow
	for i in range(nF):
		im1 = iframes[i]
		fn1 = int(os.path.splitext(os.path.basename(im1))[0].split('_')[1])
		for j in range(i+1,nF):
			im2 = iframes[j]
			fn2 = int(os.path.splitext(os.path.basename(im2))[0].split('_')[1])
			print("DeepFlow between frame %d and %d" %(fn1, fn2))
			flow_in1 = args.path_in + 'corrmatrix/%04d_%04d.flo'%(fn1,fn2)
			flow_in2 = args.path_in + 'corrmatrix/%04d_%04d.flo'%(fn2,fn1)

			#Read in flow
			flow1 = readFlo(flow_in1)
			flow2 = readFlo(flow_in2)

			ny,nx = flow1.shape[0:2]

			#For each run we compute the average reconstruction error
			fwdmeshy, fwdmeshx = [a.astype(np.float32) for a in np.meshgrid(np.arange(nx), np.arange(ny))]
			#Perturb mesh grid by forward flow 
			#Round to integers 
			fwdx = fwdmeshx + np.ceil(flow1[:,:,0])
			fwdy = fwdmeshy + np.ceil(flow1[:,:,1])
			fwdx = np.maximum(0, np.minimum(nx-1, fwdx))
			fwdy = np.maximum(0, np.minimum(nx-1, fwdy))
			#Look up flow field using this perturbed map
			fwdremapx = fwdx + flow2[fwdx.astype(int),fwdy.astype(int),0]
			fwdremapy = fwdy + flow2[fwdx.astype(int),fwdy.astype(int),1]
			fwdremapx -= fwdmeshx 
			fwdremapy -= fwdmeshy 
			fwderr = np.sqrt(fwdremapx**2 + fwdremapy**2)
			#fwdtracked = fwderr < threshold

			D[i,j] = np.mean(fwderr)
			D[j,i] = D[i,j]

	# Plot distance matrix.
	fig1 = pylab.figure(figsize=(8,8))
	axmatrix1 = fig1.add_axes([0.3,0.1,0.6,0.6])
	im = axmatrix1.matshow(D, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
	fn_out = args.path_in + 'similarity.png'
	fig1.savefig(fn_out)

	#Once we've loaded this data we view the similarity matrix
	fig = pylab.figure(figsize=(8,8))
	ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
	Y = sch.linkage(D, method='centroid')
	Z1 = sch.dendrogram(Y, orientation='right')
	ax1.set_xticks([])
	ax1.set_yticks([])
	
	# Compute and plot second dendrogram.
	ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
	Y = sch.linkage(D, method='single')
	Z2 = sch.dendrogram(Y)
	ax2.set_xticks([])
	ax2.set_yticks([])
	
	# Plot distance matrix.
	axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
	idx1 = Z1['leaves']
	idx2 = Z2['leaves']
	D = D[idx1,:]
	D = D[:,idx2]
	im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
	axmatrix.set_xticks([])
	axmatrix.set_yticks([])
	
	# Plot colorbar.
	axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
	pylab.colorbar(im, cax=axcolor)
	#fig.show()
	fn_out = args.path_in + 'dendrogram.png'
	fig.savefig(fn_out)

	#Make another version of this plot but bigger and with frame snippets
	#Load all the iframe images and resize
	fn_out_d1 = args.path_in + 'dend_d1_tile.png'
	fn_out_d2 = args.path_in + 'dend_d2_tile.png'
	im_d1 = images[idx1[0]]
	im_d2 = images[idx2[0]]
	for idx in range(1,nF):
		im_d1 = np.hstack((im_d1, images[idx1[idx]]))
		im_d2 = np.hstack((im_d2, images[idx2[idx]]))
	cv2.imwrite(fn_out_d1,im_d1)
	cv2.imwrite(fn_out_d2,im_d2)


if __name__ == "__main__":
	sys.exit(main())
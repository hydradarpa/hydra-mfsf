#!/usr/bin/env pythonx
import sys 
import argparse 
from stitcher import Stitcher
from scipy.io import loadmat, savemat 
import numpy as np 

from matplotlib import pyplot as plt

from vispy import gloo
from vispy import app

#Test code
class Args:
	pass 
args = Args()
args.fn_out = './test_stitch_smalldata.mat'
args.flow_in = ['./mfsf_output/stack0001_nref100_nframe250/result.mat',\
				'./mfsf_output/stack0002_nref100_nframe250/result.mat']

nV = len(args.flow_in)
if nV != 2:
	print("Specify more than 1 MFSF results file")
	#return 

nref = np.zeros(nV)
nF = np.zeros(nV)

#Load first video and find last frame's coordinates
a = loadmat(args.flow_in[0])
params = a['parmsOF']

u0 = a['u']
v0 = a['v']
info = a['info']
nF[0] = u0.shape[2]	
nx = u0.shape[0]
ny = u0.shape[1]

#Load remainder of MFSF data to get total frames
for vidx in range(1,nV):
	a = loadmat(args.flow_in[vidx])	
	nF[vidx] = a['u'].shape[2]	

#Make some artificial data
nF[0] = 10; nF[1] = 10
nx = 10
ny = 15
u0 = np.zeros((ny, nx, nF[0]))
v0 = np.zeros((ny, nx, nF[0]))
for i in range(nF[0].astype(int)):
	u0[:,:,i] = i/5.
	v0[:,:,i] = 2*i/5.

#Initialize data
us = np.zeros((ny, nx, np.sum(nF)))
vs = np.zeros((ny, nx, np.sum(nF)))
us[:,:,0:nF[0]] = u0
vs[:,:,0:nF[0]] = v0

#Load in optic flow data
vidx = 1
#Load MFSF data
u1 = np.zeros((ny, nx, nF[vidx]))
v1 = np.zeros((ny, nx, nF[vidx]))
for i in range(nF[0].astype(int)):
	v1[:,:,i] = -i/5.
#Make a Stitcher
thestitch = Stitcher(u1, v1)
self = thestitch
(u, v) = thestitch.run(u0, v0)

us[:,:,np.sum(nF[0:vidx]):np.sum(nF[0:vidx+1])] = u
vs[:,:,np.sum(nF[0:vidx]):np.sum(nF[0:vidx+1])] = v

#Save output matrix
#mdict = {'u':us, 'v':vs, 'parmsOF':params, 'info':info}
#savemat(args.fn_out, mdict)

#!/usr/bin/env python
import sys, argparse 
from kalman import KalmanFilter
from renderer import VideoStream, FlowStream
from distmesh_dyn import DistMesh

import logging

import pdb 

def main():
	usage = """stitch_mfsf.py [output_matfile] [input_matfile 1] [input_matfile 2] <input_matfile 3> ...

Stitch together separate MFSF optic flow fields from separate videos into the one flow field 
whose coordinates are relative to the reference frame in the first video. This can be used for
tracking objects marked in the first video through later videos.

Example: 
./stitch_mfsf.py [stk_1-2.mat] [results_stk_0001.mat] [results_stk_0002.mat]

For help:
./stitch_mfsf.py -h 

Ben Lansdell
10/12/2016
"""

	parser = argparse.ArgumentParser()
	parser.add_argument('fn_out', help='output mat file')
	parser.add_argument('flow_in', help='input mat files from MFSF', nargs = '+')
	args = parser.parse_args()

	nV = len(args.flow_in)
	if nV < 2:
		print("Specify more than 1 MFSF results file")
		return 

	nref = np.zeros(nV)
	nF = np.zeros(nV)
	nx = 0

	#Load first video and find last frame's coordinates
	a = loadmat(args.flow_in[v])
	params = a['parmsOF']
	u = a['u']
	v = a['v']
	nF[0] = u.shape[2]	
	nx = u.shape[0]
	ny = u.shape[1]

	#Quite large....
	shiftu = np.zeros((nx, ny, total_frames))
	shiftv = np.zeros((nx, ny, total_frames))
	shiftu[:,:,0:nF[0]] = u
	shiftv[:,:,0:nF[0]] = v
	offsetu_last = u[:,:,nF[0]]
	offsetv_last = v[:,:,nF[0]]
	frame_count = nF[0]

	#Load subsequent videos, offset these by last video's last frame
	#Take note of current video's last frame coordinates
	for vidx in range(1,nV):
		#Load MFSF data
		a = loadmat(args.flow_in[vidx])
	
		params = a['parmsOF']
		u = a['u']
		v = a['v']
		nF[vidx] = u.shape[2]	

		offsetu_first = u[:,:,0]
		offsetv_first = v[:,:,0]

		shiftu[:,:,frame_count:(frame_count+nF[vidx])] = u - offsetu_first + offsetu_last
		shiftv[:,:,frame_count:(frame_count+nF[vidx])] = v - offsetv_first + offsetv_last

		offsetu_last = u[:,:,-1]
		offsetv_last = v[:,:,-1]

	#Save output matrix
		
if __name__ == "__main__":
	sys.exit(main())

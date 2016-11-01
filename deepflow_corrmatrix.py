#!/usr/bin/env python
import sys 
import argparse 
import os.path 
import os

import numpy as np 
from glob import glob 

from cvtools import readFlo

import matplotlib.pyplot as plt
import cv2 

def main():
	usage = """deepflow_corrmatrix.py [register_dir]...

Generate confidence maps based on optical flow estimation

Example: 
./deepflow_corrmatrix.py ./register/20160412stk0001/

For help:
./deepflow_corrmatrix.py -h 

Ben Lansdell
10/30/2016
"""

	parser = argparse.ArgumentParser()
	parser.add_argument('dir_in', help='output directory')
	args = parser.parse_args()

	#Test code for interactive dev
	class Args:
		pass 
	args = Args()
	args.dir_in = './register/20160412stk0001/'

	#Get the set of reference frames...
	refframes = [int(i[-8:-4]) for i in glob(args.dir_in + 'refframes/*.png')]
	nF = len(refframes)

	threshold = 4
	radius = 6

	#Load DeepFlow results
	for r1 in refframes:
		for r2 in refframes:
			if r1 != r2:

				print("Load DeepFlow between frame %d and %d" %(r1, r2))
				fn_in1 = args.dir_in + 'corrmatrix/%04d_%04d.flo'%(r1, r2)
				fn_in2 = args.dir_in + 'corrmatrix/%04d_%04d.flo'%(r2, r1)

				flow1 = readFlo(fn_in1)
				flow2 = readFlo(fn_in2)

				nx = flow1.shape[0]
				ny = flow1.shape[1]

				#Flip x and y flow
				flow1 = np.transpose(flow1, [1,0,2])
				flow2 = np.transpose(flow2, [1,0,2])
				flow1 = flow1[:,:,::-1]
				flow2 = flow2[:,:,::-1]

				#Perform mapping and then reverse mapping, then perform reverse mapping then mapping
				#Make mesh grid
				fwdmeshx, fwdmeshy = [a.astype(np.float32) for a in np.meshgrid(np.arange(nx), np.arange(ny))]
				revmeshx, revmeshy = [a.astype(np.float32) for a in np.meshgrid(np.arange(nx), np.arange(ny))]

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

				#Determine based on absolute value of displacement whether 'mapped' or not...
				#perhaps model as GMM/some sort of smoothing....
				#plt.pcolor(fwderr)
				#plt.colorbar()
				#plt.show()

				#Plot flow data 
				#absflow1 = np.sqrt(flow1[:,:,0]**2 + flow1[:,:,1]**2)
				#absflow2 = np.sqrt(flow2[:,:,0]**2 + flow2[:,:,1]**2)

				#Load reference frame
				fn_in1 = args.dir_in + 'refframes/frame_%04d.png'%r1
				rf1 = cv2.imread(fn_in1)
				fn_in2 = args.dir_in + 'refframes/frame_%04d.png'%r2
				rf2 = cv2.imread(fn_in2)

				hsv = np.zeros_like(rf1)
				hsv[...,1] = 255
				mag, ang = cv2.cartToPolar(flow1[...,0], flow1[...,1])
				hsv[...,0] = ang*180/np.pi/2
				hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
				bgr1 = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
				dst1 = cv2.addWeighted(rf1,0.7,bgr1,0.3,0)
				#cv2.imshow('f',dst1)

				hsv = np.zeros_like(rf2)
				hsv[...,1] = 255
				mag, ang = cv2.cartToPolar(flow2[...,0], flow2[...,1])
				hsv[...,0] = ang*180/np.pi/2
				hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
				bgr2 = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
				dst2 = cv2.addWeighted(rf2,0.7,bgr2,0.3,0)
				#cv2.imshow('f',dst2)

				#Overlay with reference frame...
				im_fwderr = cv2.normalize(fwderr, fwderr, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
				im_fwderr = cv2.cvtColor(im_fwderr, cv2.COLOR_GRAY2BGR)
				im_fwderr = cv2.applyColorMap(im_fwderr, cv2.COLORMAP_JET)
				dst = cv2.addWeighted(rf1,0.7,im_fwderr,0.3,0)

				#cv2.imshow('Error', dst)

				#Concat into one big image...
				vis1 = np.concatenate((rf1, rf2, bgr1), axis=1)
				vis2 = np.concatenate((rf2, rf1, bgr2), axis=1)
				vis3 = np.concatenate((bgr1, bgr2, im_fwderr), axis=1)
				vis = np.concatenate((vis1, vis2, vis3), axis = 0)
				cv2.imshow('Error', vis)

				#Other measures of error include just comparing the flow mapped R1 to R2
				#and vice versa
				#Can use CV's remap function

				rf1_recon = cv2.remap(rf2, fwdmeshx + flow1[:,:,0], fwdmeshy + flow1[:,:,1], cv2.INTER_LINEAR)
				rf2_recon = cv2.remap(rf1, revmeshx + flow2[:,:,0], revmeshy + flow2[:,:,1], cv2.INTER_LINEAR)

				#Compare these reconstructions to the original frames....
				vis1 = np.concatenate((rf1, rf1_recon), axis = 1)
				vis2 = np.concatenate((rf2, rf2_recon), axis = 1)
				vis = np.concatenate((vis1, vis2), axis = 0)
				cv2.imshow('Recon', vis)

				cv2.imwrite('./test.png', vis)

				#Run local comparisons within a window...
				sim1 = np.zeros((nx,ny))
				sim2 = np.zeros((nx,ny))
				for i in range(radius, nx-radius):
					for j in range(radius, ny-radius):
						#Generate window around current point
						pts1 = rf1[i-radius:i+radius, j-radius:j+radius]
						pts2 = rf1_recon[i-radius:i+radius, j-radius:j+radius]
						pts1 = np.reshape(pts1, (-1,1))
						pts2 = np.reshape(pts2, (-1,1))
						#Measure similiarty here
						sim1[i,j] = np.corrcoef(pts1, pts2)

						#Generate window around current point
						pts1 = rf2[i-radius:i+radius, j-radius:j+radius]
						pts2 = rf2_recon[i-radius:i+radius, j-radius:j+radius]
						pts1 = np.reshape(pts1, (-1,1))
						pts2 = np.reshape(pts2, (-1,1))
						#Measure similiarty here
						sim2[i,j] = np.corrcoef(pts1, pts2)

				#Save these results


				os.system(DF + ' %s %s %s -match %s' %(im2, im1, fn_out, matches)) 

if __name__ == "__main__":
	sys.exit(main())

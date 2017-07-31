#!/usr/bin/env python
import sys 
import argparse 
from scipy.io import loadmat, savemat 
import numpy as np 
import os
import gc 
import cv2
import glob

def main(args):
	usage = """Will run through entire pipeline. Can take a while to run, select which parts to do...

Will do the following, given [tiff_file] and [name]:
1.. Split a large tiff into 250 frame parts, save in ../hydra/video/[name] by default
2.. Run MFSF method, forward and backward. (Add Ryan's background subtraction methods here) Save in ./mfsf_output/[name]
3. Save interframes
4. Have user select the reference frames to use
5. Run DeepMatching and DeepFlow between all interframes and the reference frames
6. Compute MFSF error terms
7. Perform MS image segmentation, using combination of MFSF derived error terms and DM error terms
8. Perform continuation (currently needs OpenGL, annoying to use remotely)
9. 
10. Incorporate into particle tracker...

Ben Lansdell
5/18/2017
"""

	#Test code
	class args:
		pass
	args.name = '20140829'
	args.vid_dir = '../hydra/video/'
	args.mfsf_dir = './mfsf_output'
	args.res_dir = './simmatrix'
	args.nframes = 50
	args.refframes = '-1'
	args.refframes = [int(a) for a in args.refframes.split(',')]

	#Steps 1 and 2.
	print '** Split large tiff into %d frame stacks. Convert to 8 bit. Run MFSF forward and backward'%args.nframes
	cmd = "matlab -r \"try splitlargetiff('%s', '%s', %d); catch; display('Failed'); end; quit;\""%(args.name,args.vid_dir, args.nframes)
	cmd = "matlab -r \"splitlargetiff('%s', '%s', %d); quit;\""%(args.name,args.vid_dir, args.nframes)
	os.system(cmd)

	#Step 3.
	#Copy interframes into their own directory
	print '** Copy interframes into their own directory'
	dr = args.res_dir + '/' + args.name + '/refframes/'
	if not os.path.exists(dr):
	    os.makedirs(dr)
	nstacks = len(glob.glob(args.vid_dir + '/' + args.name + '/stk_*'))
	maxfrm = nstacks*args.nframes + 1
	if maxfrm > 10000:
		l = 5
	else:
		l = 4
	for idx in range(nstacks):
		if l == 5:
			frm = '%s/frame_%05d.tif'%(dr, idx*args.nframes+1)
		else:
			frm = '%s/frame_%04d.tif'%(dr, idx*args.nframes+1)
		src = args.vid_dir + '/' + args.name + '/stk_%04d/frames8/frame_0001.tif'%(idx+1)
		if os.path.exists(src):
			cmd = 'cp ' + src + ' ' + frm
			os.system(cmd)

	#Step 4. 
	#If refframes are selected then proceed, otherwise, ask for input
	print("** Check if reference frames have been selected by user")
	if len(args.refframes) == 1 and args.refframes[0] == -1:
		print "Please provide list of reference frames as optional argument input to proceed"
		return

	#Step 5. 
	print("** Running DeepMatching and DeepFlow between interframes and reference frames")
	if l == 5:
		refframes = ' '.join(['frame_%05d.tif'%a for a in args.refframes])
	else:
		refframes = ' '.join(['frame_%04d.tif'%a for a in args.refframes])
	cmd = './deepflow_simmatrix.py %s %s'%(args.res_dir + '/' + args.name + '/', refframes)
	#./deepflow_simmatrix.py ./simmatrix/20160412/  frame_0001.tif frame_0501.tif frame_1001.tif frame_1501.tif
	os.system(cmd)

	#Visualize
	cmd = './deepflow_viz.py %s %s'%(args.res_dir + '/' + args.name + '/', refframes)
	os.system(cmd)

	#Compute the DM error terms
	cmd = './deepflow_corrmatrix.py %s'%(args.res_dir + '/' + args.name + '/')
	os.system(cmd)

	# Step 6
	print("** Computer MFSF forward and backward error terms, all possible paths, and then segment")

	#Probably needs some modifying from previous version...
	cmd = './mfsf_corrmatrix.py '
	os.system(cmd)

	# Step 7
	print("** Continue paths and visualize")

	#Load MS segmenting results for each reference frame
	#u_s = np.load(path_in)


if __name__ == '__main__':
	#name = '20170219'
	parser = argparse.ArgumentParser()
	parser.add_argument('name', help='name of video to analyse. video must be tiff file placed in [vid_dir]/[name]/[name].tif')
	parser.add_argument('--vid_dir', help='', type = str, default='../hydra/video/')
	parser.add_argument('--mfsf_dir', help='', type = str, default='./mfsf_output/')
	parser.add_argument('--res_dir', help='', type = str, default='./simmatrix/')
	parser.add_argument('--nframes', help='', type = int, default=250)
	parser.add_argument('--refframes', help='', type = str, default='-1')
	args = parser.parse_args()
	args.refframes = [int(a) for a in args.refframes.split(',')]
	main(args)

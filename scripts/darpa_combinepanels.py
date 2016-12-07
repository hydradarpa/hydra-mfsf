#!/usr/bin/env python
import sys, os, cv2
from renderer import VideoStream
import numpy as np 
from glob import glob 

fn_in = '../hydra/video/20160412/stk_0001_0002.avi'
mfsf_in = './mfsf_output/stk_0001_mfsf_nref100/'
warp_in = './figures/warp/warp*.png'
tracked_mesh_in = './mesh_neurons/frame_????_tracked.png'
untracked_mesh_in = './mesh_neurons/frame_????_untracked.png'
tracked_errors_in = './mesh_neurons/frame_tracked_errors_????.png'
untracked_errors_in = './mesh_neurons/frame_untracked_errors_????.png'
imageoutput = mfsf_in + '/darpa_frames/'

threshold = 2

#Make directory if needed...
if not os.path.exists(imageoutput):
    os.makedirs(imageoutput)

#Pattern match to get actual number of frames used
warpfiles = sorted(glob(mfsf_in + warp_in))
trackedmeshfiles = sorted(glob(mfsf_in + tracked_mesh_in))
trackederrsfiles = sorted(glob(mfsf_in + tracked_errors_in))
untrackedmeshfiles = sorted(glob(mfsf_in + untracked_mesh_in))
untrackederrsfiles = sorted(glob(mfsf_in + untracked_errors_in))
capture = VideoStream(fn_in, threshold)
nF = len(warpfiles)
#nx = int(capture.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

nS = 1

ny = 1062
nx = 1670

vidloc = (32,50)
vidsz = (512, 512)
warploc = (64+512, 50)
warpsz = (512, 512)
meshloc = (96+512*2, 50)
meshsz = (512, 512)
errsloc = (286, 580)

errsim = cv2.imread(trackederrsfiles[0])
errsy = int(errsim.shape[0]*1136./errsim.shape[1])
errssz = (1136,errsy)

for idx in range(nF):
	image = 255*np.ones((ny, nx, 3))
	print("Visualizing frame %d" % idx)
	#Read in data
	ret, frame, grayframe, mask = capture.read()
	warpim = cv2.imread(warpfiles[idx])
	meshim = cv2.imread(trackedmeshfiles[idx])
	errsim = cv2.imread(trackederrsfiles[idx])

	#Resize input images
	frame = cv2.resize(frame, vidsz)
	warpim = cv2.resize(warpim, warpsz)
	meshim = cv2.resize(meshim, meshsz)
	errsim = cv2.resize(errsim, errssz)

	#Compose image
	image[vidloc[1]:(vidloc[1]+vidsz[1]), vidloc[0]:(vidloc[0]+vidsz[0])] = frame
	image[warploc[1]:(warploc[1]+warpsz[1]), warploc[0]:(warploc[0]+warpsz[0])] = warpim
	image[meshloc[1]:(meshloc[1]+meshsz[1]), meshloc[0]:(meshloc[0]+meshsz[0])] = meshim
	image[errsloc[1]:(errsloc[1]+errssz[1]), errsloc[0]:(errsloc[0]+errssz[0])] = errsim

	#Save image
	fn_out = "%s/frame_%04d.png"%(imageoutput,idx)
	cv2.imwrite(fn_out, image)

#Write some static frames of the last image
for idx in range(nF, nF+nS):
	fn_out = "%s/frame_%04d.png"%(imageoutput,idx)
	cv2.imwrite(fn_out, image)

#Write the images with untracked points
for idx in range(nF):
	image = 255*np.ones((ny, nx, 3))
	print("Visualizing frame %d" % idx)
	#Read in data
	ret, frame, grayframe, mask = capture.read()
	warpim = cv2.imread(warpfiles[idx])
	meshim = cv2.imread(untrackedmeshfiles[idx])
	errsim = cv2.imread(untrackederrsfiles[idx])

	#Resize input images
	frame = cv2.resize(frame, vidsz)
	warpim = cv2.resize(warpim, warpsz)
	meshim = cv2.resize(meshim, meshsz)
	errsim = cv2.resize(errsim, errssz)

	#Compose image
	image[vidloc[1]:(vidloc[1]+vidsz[1]), vidloc[0]:(vidloc[0]+vidsz[0])] = frame
	image[warploc[1]:(warploc[1]+warpsz[1]), warploc[0]:(warploc[0]+warpsz[0])] = warpim
	image[meshloc[1]:(meshloc[1]+meshsz[1]), meshloc[0]:(meshloc[0]+meshsz[0])] = meshim
	image[errsloc[1]:(errsloc[1]+errssz[1]), errsloc[0]:(errsloc[0]+errssz[0])] = errsim

	#Save image
	fn_out = "%s/frame_%04d.png"%(imageoutput,idx+nF+nS)
	cv2.imwrite(fn_out, image)

#Make movie
avconv = 'avconv -r 16 -i ' + imageoutput + '/frame_%04d.png -c:v h264 -r 16 -qscale 1 -y'
os.system(avconv + ' ' + imageoutput + '/output.mp4')

#!/usr/bin/env python
import sys 
import os

import numpy as np 
import cv2
import matplotlib.pyplot as plt 

from lib.renderer import VideoStream

from cvtools import readFileToMat

flow_in = '../hydra/video/20160412/combined/flow/combined' 
vid_in = '../hydra/video/20160412/combined/combined.avi'
vid_in_lossy = '../hydra/video/20160412/combined/combined_lossy.avi'
name = '20160412'
distance_in = './simmatrix/' + name + '/similarity.png'

#Make a video
print 'Making movie'
output = './simmatrix/' + name + '/simdetails/'
if not os.path.exists(output):
    os.makedirs(output)

dpi = 96

#Extract ffprobe h264 iframes
#ffprobe = "ffprobe -select_streams v -show_frames -show_entries frame=pict_type -of csv %s | grep -n I | cut -d ':' -f 1"%vid_in_lossy
#os.system(ffprobe)

#Obtained from running 
iframes = [1,120,229,321,423,672,921,968,1148,1316,1474,1582,1831,1906,2155,2222,2471,2541,2597,2787,2937,3056,3164,3413,3662,3722,3971,4029,4213,4350,4467,4716,4965]

capture = VideoStream(vid_in_lossy, 1)

nF = int(capture.cap.get(cv2.CAP_PROP_FRAME_COUNT))

nx = 512 
ny = 512

mag_nx = 1000
mag_ny = 450

magloc = (12,532)
vidloc = (0,0)
distloc = (512,0)

distance = cv2.resize(cv2.imread(distance_in),(ny,nx))

#Optic flow magnitude per time 
flowmag = np.zeros(nF)

#Change in brightness per time 
brightness = np.zeros(nF)

fig1 = plt.figure(figsize = (1*mag_nx/dpi, 1*mag_ny/dpi), dpi = dpi)
fig1.set_facecolor((1,1,1))
ax1 = fig1.add_subplot(211)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

ax2 = fig1.add_subplot(212)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

#Combine these with frame snippets/video 
for idx in range(nF):
	print 'Loading frame', idx
	ret, frame, mask, _ = capture.read()
	frame = cv2.resize(frame, (ny,nx))
	brightness[idx] = np.mean(frame)

	#Load in optic flow data and compute average magnitude
	mx = flow_in + '_%03d_x.mat'%idx
	my = flow_in + '_%03d_y.mat'%idx

	#Read as much as we can
	try:
		flowx = readFileToMat(mx)
		flowy = readFileToMat(my)
	except IOError:
		pass
		#break 
	#Compute magnitude and take its mean 
	mag = np.mean(np.sqrt(flowx*flowx + flowy*flowy))
	flowmag[idx] = mag 

	ax1.plot(np.arange(idx), flowmag[0:idx], linewidth = 3, color = (0.3,0.9,0), zorder = 1)
	ax2.plot(np.arange(idx), brightness[0:idx], linewidth = 3, color = (0.3,0.9,0), zorder = 1)
	ax1.plot(np.arange(nF)/10., flowmag, linewidth = 3, color = (0.3,0.3,0.3), zorder = 0)
	ax2.plot(np.arange(nF)/10., brightness, linewidth = 3, color = (0.3,0.3,0.3), zorder = 0)
	#ax1.set_ylim([0, 30])
	ax1.set_xlim([0, nF/10.])
	ax1.set_ylabel('|u,v| (pixels)')
	ax2.set_ylabel('brightness')
	ax2.set_xlabel('time (s)')
	ax2.set_xlim([0, nF/10.])
	#ax1.set_xticks(np.array([10, 20]))
	#ax1.set_yticks(np.array([0, 15, 30]))

	#Or just save it and read it in again...
	plt.savefig(output + '/tmp.png', facecolor=fig1.get_facecolor(), transparent=True)
	pltdata = cv2.imread(output + '/tmp.png')

	#fig1.canvas.draw()
	# Now we can save it to a numpy array.
	#pltdata = np.fromstring(fig1.canvas.tostring_rgb(), dtype=np.uint8, sep='')
	#pltdata = pltdata.reshape(fig1.canvas.get_width_height()[::-1] + (3,))

	#Compose the frame
	image = (255*np.ones((1024,1024,3))).astype(np.uint8)
	pltdata = cv2.resize(pltdata, (mag_nx,mag_ny))

	#Compose image
	image[vidloc[1]:(vidloc[1]+ny), vidloc[0]:(vidloc[0]+nx),:] = frame
	image[distloc[1]:(distloc[1]+ny), distloc[0]:(distloc[0]+nx),:] = distance
	image[magloc[1]:(magloc[1]+mag_ny), magloc[0]:(magloc[0]+mag_nx),:] = pltdata

	#Save frame 
	fn_out = output + 'frame_%04d.png'%idx
	cv2.imwrite(fn_out, image)

#Make the movie out of the frames...

avconv = 'avconv -i ' + output + 'frame_%03d.png -c:v huffyuv -y'
os.system(avconv + ' ' + output + 'output.avi')
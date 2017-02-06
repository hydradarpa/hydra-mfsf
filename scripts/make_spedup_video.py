#!/usr/bin/env python
import sys, os

imageoutput = mfsf_in + '/darpa_frames/'

#Make directory if needed...
if not os.path.exists(imageoutput):
    os.makedirs(imageoutput)

#Make movie
avconv = 'avconv -r 16 -i ' + imageoutput + '/frame_%04d.png -c:v h264 -r 16 -qscale 1 -y'
os.system(avconv + ' ' + imageoutput + '/output.mp4')

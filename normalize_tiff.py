#!/usr/bin/env python
import sys 
import argparse 

import numpy as np 
import libtiff as lt 
from libtiff import TIFF
from skimage.io import imread, imsave

def main():
	usage = """normalize_tiff.py [input_tiff] [output_tiff]

Normalize intensity of tiff stack 

Example: 
./normalize_tiff.py stk_0001.tif stk_0001_normalize.tif

For help:
./normalize_tiff.py -h 

Ben Lansdell
10/13/2016
"""

	parser = argparse.ArgumentParser()
	parser.add_argument('fn_in', help='input tiff file')
	parser.add_argument('fn_out', help='output tiff file')
	args = parser.parse_args()

	#Test code
	#class Args:
	#	pass 
	#args = Args()
	#args.fn_in = '../hydra/video/20160412/stk_0003_Substack (1-5000).tif'
	#args.fn_out = '../hydra/video/20160412/stk_0003_normalized.tif'

	# to open a tiff file for reading:
	#tif = TIFF.open(args.fn_in, mode='r')
	#Get the min/max
	#tif = imread(args.fn_in)
	#for image in tif.iter_images(): # do stuff with image
	#	mini = min(mini, np.min(image))
	#	maxi = max(maxi, np.max(image))
	#tif.close()
	#maxi *= 2

	tif = imread(args.fn_in)
	mini = np.min(tif)
	maxi = np.max(tif)
	uint8 = 2**8-1

	# to open a tiff file for writing:
	#tif = TIFF.open(args.fn_in, mode='r')
	#tif_out = TIFF.open(args.fn_out, mode='w')
	#scaled_image = np.zeros(image.shape, dtype = np.float32)

	print 'writing'
	scaled_image = (tif.astype(np.float32)-mini)/(maxi-mini)*uint8
	#tif_out.write_image(scaled_image.astype(np.uint8))
	imsave(args.fn_out, scaled_image.astype(np.uint8))

	#tif.close()
	#tif_out.close()

if __name__ == "__main__":
	sys.exit(main())

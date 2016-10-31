#!/usr/bin/env python
import sys 
import argparse 

import numpy as np 
import libtiff as lt 
from libtiff import TIFF

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
	class Args:
		pass 
	args = Args()
	args.fn_in = '../hydra/video/20160412/stk_0001_Substack (1-5000).tif'
	args.fn_out = '../hydra/video/20160412/stk_0001_normalized.tif'

	# to open a tiff file for reading:
	tif = TIFF.open(args.fn_in, mode='r')
	#Get the min/max
	for image in tif.iter_images(): # do stuff with image
		mini = np.min(image)
		maxi = np.max(image)
	tif.close()

	uint8 = 2**8-1

	# to open a tiff file for writing:
	tif = TIFF.open(args.fn_in, mode='r')
	tif_out = TIFF.open(args.fn_out, mode='w')
	scaled_image = np.zeros(image.shape, dtype = np.float32)

	for image in tif.iter_images():
		print 'writing'
		scaled_image = (image.astype(np.float32)-mini)/(maxi-mini)*uint8
		tif_out.write_image(scaled_image.astype(np.uint8))

	tif.close()
	tif_out.close()

if __name__ == "__main__":
	sys.exit(main())

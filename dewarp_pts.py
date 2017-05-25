#!/usr/bin/env python
import sys 
import argparse 
from scipy.io import loadmat, savemat 
import numpy as np 
import os
import gc 
import cv2
import glob
from lib.warper import Warper

def main(args):
	usage = """Dewarp a set of points according to MFSF estimation

	Ben Lansdell
	5/25/2017
	"""
	#Test code:
	class Args:
		pass
	#Format: ID, frame, x, y
	args.pts = './tracks/20160412/20160412_dupreannotation_stk0001.csv'
	args.output = './tracks/20160412/20160412_dupreannotation_stk0001_dewarp.csv'
	args.name = 'stack0001_nref100_nframe250'

	#Load points

	#Create Warper

	#For each frame update warping object and dewarp points

	#Add to array

	#Save as new csv

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('name', help='name of MFSF data to use. Must be found in [mfsf_dir]/[name]/result.mat')
	parser.add_argument('pts', help='file with set of points to correct. CSV with format: frame,x,y')
	parser.add_argument('output', help='output file name')
	parser.add_argument('--mfsf_dir', help='', type = str, default='./mfsf_output/')
	args = parser.parse_args()
	main(args)
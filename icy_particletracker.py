#!/usr/bin/env python
import sys, os
import argparse

import xml.etree.ElementTree as ET

from os.path import abspath 
from glob import glob 

from xlrd import open_workbook
import cPickle as pickle

import numbers 

def parseSheet(sheet):
	tracks = []
	currenttrack = []
	nR = sheet.nrows
	for r in range(nR):
		row = sheet.row(r)
		if len(row[0].value) > 0:		
			#Start a new track
			if len(currenttrack):
				tracks.append(currenttrack)
			currenttrack = []
		elif isinstance(row[2].value, numbers.Number):
			#Add to current track
			x = row[3].value
			y = row[4].value
			t = row[2].value
			pt = [x, y, t]
			currenttrack.append(pt)
	return tracks 

def main():
	usage = """icy_particletracker.py [frames_in] [fn_out] -maxfiles MAXFILES

	Run particle tracking method on series of tif files using bioimage analysis
	program Icy. Calls program at the command line. 
		
	Ben Lansdell
	02/22/2017
	"""
	
	icypath = '/home/lansdell/local/icy2/'
	protocol = 'spottracking_headless.xml'

	parser = argparse.ArgumentParser()
	parser.add_argument('frames_in', help='path to .tif files, will use all tif files in folder')
	parser.add_argument('fn_out', help='output filename for python dictionary of tracks')
	parser.add_argument('-maxfiles', help='maximum number of tif files to read', default=250)
	args = parser.parse_args()
	fn_out = args.fn_out
	frames_in = args.frames_in
	fin = [abspath(p) for p in sorted(glob(frames_in + '*.tif'))]
	fin = fin[0:mf]
	frames = ':'.join(fin)

	#Edit .xml protocol with input tif files
	tree = ET.parse(icypath + protocol)
	root = tree.getroot()
	infiles = root.find('blocks').find('block').find('variables').find('input').find('variable')
	infiles.set('value', frames)
	tree.write(icypath + protocol)

	print 'Running Icy particle tracking'
	cmd = 'cd %s; java -jar icy.jar --headless --execute plugins.adufour.protocols.Protocols protocol=%s/%s'%(icypath, icypath, protocol)
	os.system(cmd)
	print '...done.'

	print 'Saving results to python dictionary'
	xls_in = frames.split(':')[0] + '_tracking.xls'
	wb = open_workbook(xls_in)
	sheet = wb.sheet_by_name('Tracks')
	tracks = parseSheet(sheet)
	pickle.dump(tracks, open(fn_out, 'wb'))

if __name__ == "__main__":
	sys.exit(main())
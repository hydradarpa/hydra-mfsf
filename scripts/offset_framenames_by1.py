import os 
from glob import glob 
n = 250

for idx,fn in enumerate(sorted(glob('frame_*.tif'), reverse = True)):
	print 'Moving', n-idx,fn
	#print 'mv ' + fn + ' ' + vid_out + 'overlay_%00d.png'%idx
	os.system('mv ' + fn + ' frame_%03d.tif'%(n-idx)) 

"""Tools for analyzing optic flow data

"""

import numpy as np 
from scipy.io import loadmat 

from cvtools import readFlo
import h5py

def load_mfsf(fn_in):
	try:
		a = loadmat(fn_in)	
		params = a['parmsOF']
		u1 = a['u']
		v1 = a['v']
	except NotImplementedError:
		print "Failed to read using loadmat, using hdf5 library to read %s"%fn_in
		f = h5py.File(fn_in,'r')
		#Note the x and y axes may need to be switched here...
		u1 = np.transpose(np.array(f.get('u')), (1,2,0))
		v1 = np.transpose(np.array(f.get('v')), (1,2,0))
	return u1,v1

def flow_err_mfsf(fn_in1, fn_in2):

	(u,v) = load_mfsf(fn_in1)
	nx = u.shape[0]
	ny = u.shape[1]
	flow1 = np.zeros((nx, ny, 2))
	flow1[:,:,0] = u[:,:,-1]
	flow1[:,:,1] = v[:,:,-1]
	
	(u,v) = load_mfsf(fn_in2)
	flow2 = np.zeros((nx, ny, 2))
	flow2[:,:,0] = u[:,:,0]
	flow2[:,:,1] = v[:,:,0]
	
	#Flip x and y flow
	#flow1 = np.transpose(flow1, [1,0,2])
	#flow2 = np.transpose(flow2, [1,0,2])
	#flow1 = flow1[:,:,::-1]
	#flow2 = flow2[:,:,::-1]
	
	#Perform mapping and then reverse mapping, then perform reverse mapping then mapping
	#Make mesh grid
	fwdmeshy, fwdmeshx = [a.astype(np.float32) for a in np.meshgrid(np.arange(nx), np.arange(ny))]
	revmeshy, revmeshx = [a.astype(np.float32) for a in np.meshgrid(np.arange(nx), np.arange(ny))]
	
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

	return fwderr 

def flow_err_mfsf_rev(fn_in1, fn_in2):

	(u,v) = load_mfsf(fn_in1)
	nx = u.shape[0]
	ny = u.shape[1]
	flow1 = np.zeros((nx, ny, 2))
	flow1[:,:,0] = u[:,:,-1]
	flow1[:,:,1] = v[:,:,-1]
	
	(u,v) = load_mfsf(fn_in2)
	flow2 = np.zeros((nx, ny, 2))
	flow2[:,:,0] = u[:,:,0]
	flow2[:,:,1] = v[:,:,0]
	
	#Flip x and y flow
	#flow1 = np.transpose(flow1, [1,0,2])
	#flow2 = np.transpose(flow2, [1,0,2])
	#flow1 = flow1[:,:,::-1]
	#flow2 = flow2[:,:,::-1]
	
	#Perform mapping and then reverse mapping, then perform reverse mapping then mapping
	#Make mesh grid
	fwdmeshy, fwdmeshx = [a.astype(np.float32) for a in np.meshgrid(np.arange(nx), np.arange(ny))]
	revmeshy, revmeshx = [a.astype(np.float32) for a in np.meshgrid(np.arange(nx), np.arange(ny))]
	
	#Perturb mesh grid by forward flow 
	#Round to integers 
	fwdx = fwdmeshx + np.ceil(flow2[:,:,0])
	fwdy = fwdmeshy + np.ceil(flow2[:,:,1])
	
	fwdx = np.maximum(0, np.minimum(nx-1, fwdx))
	fwdy = np.maximum(0, np.minimum(nx-1, fwdy))
	
	#Look up flow field using this perturbed map
	fwdremapx = fwdx + flow1[fwdx.astype(int),fwdy.astype(int),0]
	fwdremapy = fwdy + flow1[fwdx.astype(int),fwdy.astype(int),1]
	
	fwdremapx -= fwdmeshx 
	fwdremapy -= fwdmeshy 
	
	fwderr = np.sqrt(fwdremapx**2 + fwdremapy**2)

	return fwderr 

def flow_err_deepflow(fn_in1, fn_in2):
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
	fwdmeshy, fwdmeshx = [a.astype(np.float32) for a in np.meshgrid(np.arange(nx), np.arange(ny))]
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
	return fwderr 
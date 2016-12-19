#############################
#Image segmentation on CUDA #
#############################

#Lansdell 2016

import numpy as np
from jinja2 import Template 
import pdb 
import cv2

try:
	print 'Importing pycuda'
	import pycuda.driver as cuda_driver
	import pycuda
	from pycuda.compiler import SourceModule
	import pycuda.gpuarray as gpuarray
	BLOCK_SIZE = 1024
	print '...success'
except:
	print "...pycuda not installed"

def J1(x, h):
	return np.sum(np.linalg.norm(grad(x,h), axis = 2))

def chambolle(x, y, z, tau, sigma, theta, K, K_star, f, res_F, res_G, res_H, j_tv, n_iter = 100, eps = 1e-6):
	x_bar = x.copy()
	x_old = x.copy()
	print('====================================================================\nIter:\tdX:\t\tJ(u):\t\tf:\t\tPrimal objective:')
	for n in range(n_iter):
		err = np.linalg.norm(x-x_old)
		ju = j_tv(x)
		fu = np.sum(f*x)
		obj = fu + ju
		print('%d\t%e\t%e\t%e\t%e'%(n, err, ju, fu, obj))
		if (err < eps) and (n > 0):
			break
		x_old = x.copy()
		y = res_F(y + sigma*K(x_bar))
		z = res_H(z + sigma*x_bar)
		x = res_G(x - tau*(K_star(y)+z+f))
		x_bar = x + theta*(x - x_old)
	return x

def grad(u, h):
	k = u.shape[2]
	l = u.shape[3]
	p = np.zeros((u.shape[0], u.shape[1], 2, k, l))
	for j in range(l):
		for i in range(k):
			p[0:-1, :, 0, i, j] = (u[1:, :, i, j] - u[0:-1, :, i, j])/h
			p[:, 0:-1, 1, i, j] = (u[:, 1:, i, j] - u[:, 0:-1, i, j])/h
	return p 

def div(p, h):
	k = p.shape[3]
	l = p.shape[4]
	u = np.zeros((p.shape[0], p.shape[1], k, l))
	for j in range(l):
		for i in range(k):
			u[1:,:,i,j]  = (p[1:, :, 0, i, j] - p[0:-1, :, 0, i, j])/h
			u[:,1:,i,j] += (p[:, 1:, 1, i, j] - p[:, 0:-1, 1, i, j])/h
	return u 

def project_balls(p):
	#print 'Projection onto unit balls'
	pt = np.transpose(p, (0,1,3,2,4))
	n = np.linalg.norm(pt, axis = 3)
	d = np.maximum(2*n, 1)
	for i in range(pt.shape[3]):
		pt[:,:,:,i,:] = pt[:,:,:,i,:]/d
	p = np.transpose(pt, (0,1,3,2,4))
	return p 

def project_simplex(u):
	(ny, nx, k, l) = u.shape

	def proj_prob(xv):
		x = np.array(xv)
		if not len(x.shape) == 1:
			return 1.
		D = x.shape[0]
		uv = np.sort(x)[::-1]
		vv = uv + np.array([1./j - np.sum(uv[0:j])/float(j) for j in range(1,D+1)])
		rho = np.max(np.where(vv > 0))
		lmbda = (1 - np.sum(uv[0:rho+1]))/float(rho+1)
		xp = np.maximum(x + lmbda, 0)
		return xp 

	#Very slow...needs to be parallelized 
	for f in range(l):
		for i in range(ny):
			for j in range(nx):
				u[i,j,:,f] = proj_prob(np.squeeze(u[i,j,:,f]))

	return u 

#Here expecting a scalar...
def softmax(x, alpha):
	return max(x-alpha,0)

def group_LASSO(u, rho):
	prox = np.zeros(u.shape)
	k = u.shape[2]
	for i in range(k):
		ui = u[:,:,i,:]
		n = np.sqrt(np.sum(ui*ui))
		if n > 0:
			prox[:,:,i,:] = softmax(n, rho)*ui/n
	return prox
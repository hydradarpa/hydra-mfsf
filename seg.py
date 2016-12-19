#!/usr/bin/env python
import numpy as np 
import cv2 

import argparse
from os.path import basename
import os.path 
import os 

from seg_cuda import *

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return (np.array(scalar_map.to_rgba(index))*255).astype(np.uint8)[0:3]
    return map_index_to_rgb_color

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

def mumford_glasso(path_in, iframes, refframes, l=5, r = 1e4):
	#Params from [1]
	theta = 1
	tau = 0.05
	h = 1
	L2 = 8/h**2
	sigma = 1/(L2 * tau)
	lmda = l
	rho = r 				#Need to experiment with good values for rho...

	#Test params
	path_in = './register/20160412stk0001-0008/'
	iframes = [1, 501, 1001, 1501]
	refframes = [1, 501, 1001, 1501]

	nK = len(refframes)
	nL = len(iframes)
	ny = nx = 1024

	dr = path_in + './glasso_viz/'
	if not os.path.exists(dr):
	    os.makedirs(dr)

	#Generate resolvents and such
	#res_F = project_balls_intersect
	res_F = project_balls
	res_G = project_simplex
	res_H = lambda x: group_LASSO(x, rho)
	K = lambda x: grad(x, h)
	K_star = lambda x: -div(x, h)
	j_tv = lambda x: J1(x, h)

	#Generate set of images, f^{kl}, that are the error measures for each pixel
	f = np.zeros((ny, nx, nK, nL))
	#Here we load data from optic flow errors
	for l in range(nL):
		for k in range(nK):
			if l != k:
				#Load error terms
				fn_err = path_in + 'corrmatrix/%04d_%04d_deepflow_err.npz'%(refframes[k], iframes[l])
				err = np.load(fn_err)['fwderr']
				f[:,:,k,l] = lmda*err/2
				#Or should this be squared??

	#Init u, p
	u = res_G(np.zeros((ny, nx, nK, nL)))
	p = res_F(K(u))
	q = res_H(u)

	#Run chambolle algorithm
	n_iter = 30
	u_s = chambolle(u, p, q, tau, sigma, theta, K, K_star, f, res_F,\
	 res_G, res_H, j_tv, n_iter = n_iter)

	#Assign a color to each of the refframes
	cmaps = get_cmap(nK+1)
	cm = np.zeros((nK,3))
	for k in range(nK):
		cm[k,:] = cmaps(k)

	for l in range(nL):
		#Load the image for each iframe
		iframe_fn = path_in + 'refframes/frame_%04d.png'%(iframes[l])
		img = cv2.imread(iframe_fn)
		#Take argmax of u tensor to obtain segmented image
		ms_img = np.zeros(img.shape, dtype = np.uint8)
		for i in range(ny):
			for j in range(nx):
				col = np.argmax(u_s[i,j,:,l])
				ms_img[i,j,:] = cm[col,:]
		dst = cv2.addWeighted(img,0.7,ms_img,0.3,0)
		cv2.imwrite('%siframe_%d_MS_lambda_%.02e_niter_%04d.png'%(dr,l,lmda,n_iter), dst)

	return u_s
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

import scipy.ndimage 

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
		x = res_G(x - tau*(K_star(y)+f))
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

def ADMM_glasso(u, sigma, rho, n_iter = 100):
	eps_abs = 1e-6;
	eps_rel = 1e-3;
	nr_p = np.inf 
	nr_d = np.inf 
	e_p = 0 
	e_d = 0
	
	sz_p = np.prod(u.shape)
	sz_d = sz_p

	#Set ICs to zeros 
	Lambda = np.zeros(u.shape)
	v = u.copy()
	gamma = 1.0

	n = 0
	print("\titer:\t nr_p:\teps_p:\tnr_d:\teps_d")
	while (nr_p > e_p or nr_d > e_d) and (n < n_iter):
		print("\t%d\t%f\t%f\t%f\t%f"%(n, nr_p, e_p, nr_d, e_d))
		u_new = project_simplex((2*u+gamma*v-Lambda)/(2+gamma))
		v_new = group_LASSO(u_new+Lambda/gamma, rho/2/gamma)
		Lambda_new = Lambda + gamma*(u_new-v_new)

		#Compute convergence criteria
		r_p = u_new - v_new
		r_d = gamma*(v - v_new)
		e_p = np.sqrt(sz_p)*eps_abs + eps_rel*max(np.linalg.norm(u_new), np.linalg.norm(v_new))
		e_d = np.sqrt(sz_d)*eps_abs + eps_rel*np.linalg.norm(Lambda)
		nr_p = np.linalg.norm(r_p)
		nr_d = np.linalg.norm(r_d)
		
		#Heuristics to adjust dual gradient ascent rate to balance primal and
		#dual convergence
		if (nr_p > 10*nr_d):
		    gamma = 1.5*gamma
		elif nr_d > 10*nr_p:
		    gamma = gamma/1.5
		
		#Update
		u = u_new
		v = v_new
		Lambda = Lambda_new
		n += 1

	if n >= n_iter:
		print "\tADMM did not converge; reachced max iterations"
	else:
		print "\tADMM converged with:\n%d\t%f\t%f\t%f\t%f"%(n, nr_p, e_p, nr_d, e_d)

	return u 

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
		#print n 
		if n > 0:
			prox[:,:,i,:] = softmax(n, rho)*ui/n
	return prox

def mumford_glasso(path_in, iframes, refframes, l = 5, r = 1e6):
	l = 1e-4
	r = 1e-4

	#Params from [1]
	theta = 1
	tau = .5
	h = 1
	L2 = 8/h**2
	sigma = 1/(L2 * tau)
	lmda = l
	rho = r 				#Need to experiment with good values for rho...

	self_score = 9
	pitch = 5 
	radius = 10
	#Displacement scores above this will be encouraged to be labeled occluded
	occlusion_score = 30	#(in pixels)

	#Test params
	#path_in = './register/20160412stk0001-0008/'
	#iframes = [1, 501, 1001, 1501]
	#refframes = [1, 501]
	#iframes = [1, 501]
	#refframes = [1, 501]

	path_in = './register/jellyfish/'
	iframes = [1, 2, 3, 4]
	refframes = [1, 3]

	nK = len(refframes)
	nL = len(iframes)
	ny = nx = 128 

	ext = 'jpg'
	iframe_fn = path_in + 'refframes/frame_%04d.%s'%(iframes[0], ext)
	img = cv2.imread(iframe_fn)
	ny_in, nx_in = img.shape[0:2]

	dr = path_in + './glasso_viz/'
	if not os.path.exists(dr):
	    os.makedirs(dr)

	#Generate resolvents and such
	#res_F = project_balls_intersect
	res_F = project_balls
	#res_G = project_simplex
	res_G = lambda x: ADMM_glasso(x, sigma, rho)
	res_H = lambda x: group_LASSO(x, rho)
	K = lambda x: grad(x, h)
	K_star = lambda x: -div(x, h)
	j_tv = lambda x: J1(x, h)

	#Generate set of images, f^{kl}, that are the error measures for each pixel
	f = np.zeros((ny, nx, nK, nL))
	
	##Here we load data from optic flow errors
	#for l in range(nL):
	#	for k in range(nK):
	#		if refframes[k] != iframes[l]:
	#			#Load error terms
	#			fn_err = path_in + 'corrmatrix/%04d_%04d_deepflow_err.npz'%(refframes[k], iframes[l])
	#			err = np.load(fn_err)['fwderr']
	#			#Resize
	#			err = cv2.resize(err, (ny, nx))
	#			f[:,:,k,l] = lmda*err/2
	#			#Or should this be squared??

	#Instead we can load the deep matching results
	#For each match between ref frame and iframes, we add the match as a penalty
	#in all the other frames that are not the refframe.
	for l in range(nL):
		for k in range(nK):
			if refframes[k] != iframes[l]:
				#Load error terms
				fn_matches = path_in + 'corrmatrix/%04d_%04d.txt'%(refframes[k], iframes[l])
				#Load matches
				#x1 y1   x2   y2   score   ?
				#8  632  892  716  5.65289 0
				with open(fn_matches, 'r') as f_matches:
					for line in f_matches:
						(x1, y1, x2, y2, score) = [float(x) for x in line.split()[0:5]]
						(x1, y1, x2, y2) = (x1/nx_in*nx, y1/ny_in*ny, x2/nx_in*nx, y2/ny_in*ny)
						for j in np.setdiff1d(np.arange(nK), np.array([k])):
							f[int(y2),int(x2),j,l] = score
			else:
				xs,ys = np.meshgrid(np.arange(0, nx, pitch), np.arange(0, ny, pitch))
				for j in np.setdiff1d(np.arange(nK), np.array([k])):
					f[xs,ys,j,l] = self_score

	#Alternatively, we can just introduce a negative matching error. Though this introduces a non-convexity, I
	#believe... so this is a bad idea
	#for l in range(nL):
	#	for k in range(nK):
	#		if l != k:
	#			#Load error terms
	#			fn_matches = path_in + 'corrmatrix/%04d_%04d.txt'%(refframes[k], iframes[l])
	#			#Load matches
	#			#x1 y1   x2   y2   score   ?
	#			#8  632  892  716  5.65289 0
	#			with open(fn_matches, 'r') as f_matches:
	#				for line in f_matches:
	#					(x1, y1, x2, y2, score) = [float(x) for x in line.split()[0:5]]
	#					(x1, y1, x2, y2) = (x1/nx_in*nx, y1/ny_in*ny, x2/nx_in*nx, y2/ny_in*ny)
	#					f[int(y2),int(x2),k,l] = -score
	#		else:
	#			xs,ys = np.meshgrid(np.arange(0, nx, pitch), np.arange(0, ny, pitch))
	#			f[xs,ys,k,l] = -self_score

	for l in range(nL):
		for k in range(nK):
			#Filter scores to smooth a little.
			im = f[:,:,k,l]
			f[:,:,k,l] = scipy.ndimage.filters.gaussian_filter(im, radius)

	#Init u, p
	u = res_G(np.zeros((ny, nx, nK, nL)))
	p = res_F(K(u))
	q = res_H(u)

	#Run chambolle algorithm
	n_iter = 40

	#CPU
	u_s_cpu = chambolle(u, p, q, tau, sigma, theta, K, K_star, f, res_F,\
	 res_G, res_H, j_tv, n_iter = n_iter)

	#GPU
	#MSSeg = GPUChambolle(u, p, q, tau, sigma, theta, rho, f, n_iter = n_iter, eps = 1e-6)
	#u_s = MSSeg.run()

	#Assign a color to each of the refframes
	cmaps = get_cmap(nK+1)
	cm = np.zeros((nK,3))
	for k in range(nK):
		cm[k,:] = cmaps(k)

	#for l in range(nL):
	#	#Load the image for each iframe
	#	iframe_fn = path_in + 'refframes/frame_%04d.png'%(iframes[l])
	#	img = cv2.imread(iframe_fn)
	#	img = cv2.resize(img, (ny, nx))
	#	#Take argmax of u tensor to obtain segmented image
	#	ms_img = np.zeros(img.shape, dtype = np.uint8)
	#	for i in range(ny):
	#		for j in range(nx):
	#			col = np.argmax(u_s[i,j,:,l])
	#			ms_img[i,j,:] = cm[col,:]
	#	dst = cv2.addWeighted(img,0.7,ms_img,0.3,0)
	#	cv2.imwrite('%siframe_%d_MS_lambda_%.02e_niter_%04d.png'%(dr,l,lmda,n_iter), dst)

	for l in range(nL):
		#Load the image for each iframe
		iframe_fn = path_in + 'refframes/frame_%04d.%s'%(iframes[l], ext)
		img = cv2.imread(iframe_fn)
		img = cv2.resize(img, (ny, nx))
		#Take argmax of u tensor to obtain segmented image
		ms_img = np.zeros(img.shape, dtype = np.uint8)
		for i in range(ny):
			for j in range(nx):
				col = np.argmax(u_s_cpu[i,j,:,l])
				ms_img[i,j,:] = cm[col,:]
		dst = cv2.addWeighted(img,0.7,ms_img,0.3,0)
		#Add circles in bottom left with key
		for k in range(nK):
			ctr = (5, k*10+5)
			cv2.circle(dst, ctr, 3, cm[k,:], -1)

		#cv2.imwrite('%scpu_iframe_%d_MS_lambda_%.02e_niter_%04d.png'%(dr,l,lmda,n_iter), dst)
		cv2.imwrite('%scpu_iframe_%d_MS_lambda_%.02e_niter_%04d.png'%(dr,l,lmda,n_iter), dst)

	return u_s
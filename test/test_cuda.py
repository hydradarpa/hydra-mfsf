import unittest
import numpy.testing as npt 

from seg import * 
from lib.seg_cuda import * 

import numpy as np
from jinja2 import Template 
import pdb 
import cv2

import pycuda.driver as cuda_driver
import pycuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
BLOCK_SIZE = 256

class TestCUDA(unittest.TestCase):

	def setUp(self):
	
		nK = 3;
		nL = 2;
		nX = 43;
		nY = 544;
	
		x = np.zeros((nX, nY, nK, nL), dtype = np.float32)
		x[0:1,0:4,0:2,:] = 1 
		y = np.zeros((nX, nY, 2, nK, nL), dtype = np.float32)
		y[0:1,0:4,0:4,:,:] = 1 
		z = np.zeros((nX, nY, nK, nL), dtype = np.float32)
		f = np.ones((nX, nY, nK, nL), dtype = np.float32)
		tau = 1 
		sigma = 1 
		theta = 1 
		rho = 1 
	
		n_iter = 2 
		eps = 1e-6 
	
		self.width, self.height, self.nK, self.nL = x.shape
	
		self.x = x 
		self.y = y 
		self.z = z 
		self.f = f 
	
		self.tau = tau 
		self.sigma = sigma 
		self.theta = theta 
		self.rho = rho 
	
		self.n_iter = n_iter 
		self.eps = eps 
	
		self.MSSeg = GPUChambolle(x, y, z, tau, sigma, theta, rho, f, n_iter = n_iter, eps = 1e-6)
		self.cuda_tpl = self.MSSeg.cuda_tpl 
	
		cuda_source = self.cuda_tpl.render(block_size=BLOCK_SIZE, tau = self.tau,\
					sigma = self.sigma, theta = self.theta, n_iter = self.n_iter,\
					eps = self.eps, nK = self.nK)
		cuda_module = SourceModule(cuda_source, no_extern_c=1)
	
		self.cuda_glasso = cuda_module.get_function("glasso")
		#self.cuda_seg = cuda_module.get_function("segmentation")
		self.cuda_glasso.prepare("PPPiiiiii")
		#self.cuda_seg.prepare("PPPPPPiiiii")

		self.cuda_unitball = cuda_module.get_function("unitball")
		self.cuda_unitsimplex = cuda_module.get_function("project_unitsimplex")
		self.cuda_unitball.prepare("PPPiiiii")
		self.cuda_unitsimplex.prepare("PPPPPiiiii")

		self.nElements = self.width*self.height
		self.nBlocks = np.ceil(self.nElements/float(BLOCK_SIZE)).astype(int)
		self.grid_dimensions = (self.nBlocks, 1)
		self.block_dimensions = (BLOCK_SIZE, 1, 1)
	
		print "No. elements:", self.nElements
		print "No. blocks", self.nBlocks 

		self.x_gpu = gpuarray.to_gpu(self.x.copy())
		self.x_bar_gpu = gpuarray.to_gpu(self.x.copy())
		self.x_old_gpu = gpuarray.to_gpu(self.x.copy())
		print self.x_gpu.gpudata 
		print self.x_old_gpu.gpudata

		self.y_gpu = gpuarray.to_gpu(self.y)
		self.z_gpu = gpuarray.to_gpu(self.z)
		self.f_gpu = gpuarray.to_gpu(self.f)
	
	def test_glasso(self):

		z = self.z_gpu.get()
		x_bar = self.x_bar_gpu.get()

		#print x_bar 
		#print self.x 
		#print z 
		#print z_gpu 

		#Run on CPU
		z_cpu = group_LASSO(z + self.sigma*x_bar, self.rho)

		#Run on GPU
		z_gpu = self.z_gpu.get()
		ps_norm_gpu = []
		for k in range(self.nK):
			ps_norm = np.zeros((self.nBlocks,1), dtype=np.float32)
			ps_norm_gpu.append(gpuarray.to_gpu(ps_norm))
		#These can be launched separately and synchronized later
		for k in range(self.nK):
			self.cuda_glasso.prepared_call(self.grid_dimensions, self.block_dimensions,\
				self.z_gpu.gpudata, self.x_bar_gpu.gpudata, ps_norm_gpu[k].gpudata, np.uint32(k),\
				np.uint32(self.nElements), np.uint32(self.width), np.uint32(self.height),\
				np.uint32(self.nK), np.uint32(self.nL))
		cuda_driver.Context.synchronize() 

		print "GPU reduction"
		#print z_gpu 
		for k in range(self.nK):
			ps_norm = ps_norm_gpu[k].get()
			#print ps_norm 
			grp_norm = np.sqrt(np.sum(ps_norm[0:np.ceil(self.nBlocks/2.)]))
			#print grp_norm
			#Update z
			if grp_norm > 0:
				z_gpu[:,:,k,:] = (z_gpu[:,:,k,:]+self.sigma*x_bar[:,:,k,:])\
				*softmax(grp_norm, self.rho)/grp_norm

		print self.rho 
		#print z 
		#print x_bar 
		print z_cpu
		print z_gpu

		#Compare
		npt.assert_almost_equal(z_gpu, z_cpu)	
	
	#def test_seg(self):

	#	z_cpu = self.z_gpu.get()
	#	x_bar_cpu = self.x_bar_gpu.get()
	#	x_cpu = self.x_gpu.get()
	#	y_cpu = self.y_gpu.get()
	#	f_cpu = self.f_gpu.get()
	#	h = 1

	#	############
	#	#Run on CPU#
	#	############

	#	x_old_cpu = x_cpu 
	#	y_cpu = project_balls(y_cpu + self.sigma*grad(x_bar_cpu,h))
	#	x_cpu = project_simplex(x_cpu - self.tau*(div(y_cpu,h)+z_cpu+f_cpu))
	#	x_bar_cpu = x_cpu + self.theta*(x_cpu - x_old_cpu)

	#	############
	#	#Run on GPU#
	#	############

	#	self.cuda_seg.prepared_call(self.grid_dimensions, self.block_dimensions,\
	#		self.x_old_gpu.gpudata, self.x_gpu.gpudata, self.x_bar_gpu.gpudata,\
	#		self.y_gpu.gpudata, self.z_gpu.gpudata, self.f_gpu.gpudata,\
	#		np.uint32(self.nElements), np.uint32(self.width), np.uint32(self.height),\
	#		np.uint32(self.nK), np.uint32(self.nL))
	#	cuda_driver.Context.synchronize()

	#	x_old_gpu = self.x_old_gpu.get()
	#	z_gpu = self.z_gpu.get()
	#	x_bar_gpu = self.x_bar_gpu.get()
	#	x_gpu = self.x_gpu.get()
	#	y_gpu = self.y_gpu.get()

	#	#for k in range(self.nK):
	#	#	for l in range(self.nL):
	#	#		if np.isclose(y_gpu[:,:,:,k,l], y_gpu[:,:,:,k,l]).all():
	#	#			print "Not equal for k = %d, l = %d"%(k, l)
	#	#			print "GPU"
	#	#			print y_gpu[:,:,1,k,l]
	#	#			print "CPU"
	#	#			print y_cpu[:,:,1,k,l]
	#	#			print y_gpu[:,:,1,k,l]-y_cpu[:,:,0,k,l]

	#	k = l = 3
	#	print "Not equal for k = %d, l = %d"%(k, l)
	#	print "GPU"
	#	print y_gpu[:,:,1,k,l]
	#	print "CPU"
	#	print y_cpu[:,:,1,k,l]
	#	print y_gpu[:,:,1,k,l]-y_cpu[:,:,1,k,l]

	#	print np.sum(y_cpu)
	#	print np.sum(y_gpu)

	#	#Compare
	#	#npt.assert_almost_equal(x_old_gpu, x_old_cpu)	
	#	#npt.assert_almost_equal(z_gpu, z_cpu)	
	#	npt.assert_almost_equal(y_gpu, y_cpu, decimal = 7)	
	#	#npt.assert_almost_equal(x_gpu, x_cpu)	
	#	#npt.assert_almost_equal(x_bar_gpu, x_bar_cpu)	

	def test_unitball(self):

		############
		#Run on CPU#
		############

		x_bar_cpu = self.x_bar_gpu.get()
		y_cpu = self.y_gpu.get()
		h = 1
		y_cpu = project_balls(y_cpu + self.sigma*grad(x_bar_cpu,h))

		############
		#Run on GPU#
		############

		output_y = np.zeros(y_cpu.shape, dtype = np.float32)
		output_y_gpu = gpuarray.to_gpu(output_y)

		self.cuda_unitball.prepared_call(self.grid_dimensions, self.block_dimensions,\
			self.x_bar_gpu.gpudata, self.y_gpu.gpudata, output_y_gpu.gpudata,\
			np.uint32(self.nElements), np.uint32(self.width), np.uint32(self.height),\
			np.uint32(self.nK), np.uint32(self.nL))
		cuda_driver.Context.synchronize()

		y_gpu = output_y_gpu.get()

		#Compare
		npt.assert_almost_equal(y_gpu, y_cpu, decimal = 7)

	def test_unitsimplex(self):

		############
		#Run on CPU#
		############

		x_cpu = self.x_gpu.get()
		y_cpu = self.y_gpu.get()
		z_cpu = self.z_gpu.get()
		f_cpu = self.f_gpu.get()

		x_cpu_old = x_cpu.copy()
		x_cpu = project_simplex(x_cpu - self.tau*(div(y_cpu, 1)+z_cpu+f_cpu))

		############
		#Run on GPU#
		############

		output_x = np.zeros(x_cpu.shape, dtype = np.float32)
		output_x_gpu = gpuarray.to_gpu(output_x)

		self.cuda_unitsimplex.prepared_call(self.grid_dimensions, self.block_dimensions,\
			self.x_gpu.gpudata, self.y_gpu.gpudata, self.z_gpu.gpudata, self.f_gpu.gpudata,\
			output_x_gpu.gpudata, np.uint32(self.nElements), np.uint32(self.width), np.uint32(self.height),\
			np.uint32(self.nK), np.uint32(self.nL))
		cuda_driver.Context.synchronize()

		x_gpu = output_x_gpu.get()

		#x_b4 = x_cpu_old - self.tau*(div(y_cpu, 1)+z_cpu+f_cpu)
		x_b4 = x_cpu_old - self.tau*(div(y_cpu, 1)+z_cpu+f_cpu)
		#x_b4 = x_cpu_old - self.tau*(div(y_cpu, 1)+f_cpu)
		#x_b4 = f_cpu

		print "X before"
		print x_b4[:,:,0,0]
		print x_b4[:,:,1,0]
		print x_b4[:,:,2,0]
		print "GPU"
		print x_gpu[:,:,0,0]
		print x_gpu[:,:,1,0]
		print x_gpu[:,:,2,0]
		print "CPU"
		print x_cpu[:,:,0,0] 
		print x_cpu[:,:,1,0] 
		print x_cpu[:,:,2,0] 

		#Compare
		npt.assert_almost_equal(x_gpu, x_cpu, decimal = 7)

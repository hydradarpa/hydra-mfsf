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
	import pycuda.autoinit
	from pycuda.compiler import SourceModule
	import pycuda.gpuarray as gpuarray
	BLOCK_SIZE = 256
	print '...success'
except:
	print "...pycuda not installed"

#Here expecting a scalar...
def softmax(x, alpha):
	return max(x-alpha,0)

class GPUChambolle:
	def __init__(self, x, y, tau, sigma, theta, f, n_iter = 100, eps = 1e-6):

		self.width, self.height, self.nK, self.nL = x.shape

		self.x = x.astype(np.float32) 
		self.y = y.astype(np.float32) 
		self.f = f.astype(np.float32) 

		self.tau = tau 
		self.sigma = sigma 
		self.theta = theta 

		self.n_iter = n_iter 
		self.eps = eps 

		cuda_driver = pycuda.driver
		cuda_tpl = Template("""
		extern "C" {

		#include <stdio.h>

		__device__ float grad(float *x, int i, int j, int g, int k, int l, int width, int height,
					int nK, int nL) {

			unsigned int str_w = height*nK*nL;
		    unsigned int str_h = nK*nL;
		    unsigned int str_k = nL;
	    	int idx = str_w*i+str_h*j+str_k*k+l;
	    	int idx2;
		    float ret = 0.0;

		    if (g == 0)
		    {
			    if (i < width-1) {
			    	idx2 = str_w*(i+1)+str_h*j+str_k*k+l;
			    	ret = x[idx2] - x[idx];
			    } else {
			    	ret = 0.0;
			    }
		    }
		    if (g == 1)
		    {
			    if (j < height-1) {
			    	idx2 = str_w*(i)+str_h*(j+1)+str_k*k+l;
			    	ret = x[idx2] - x[idx];
			    } else {
			    	ret = 0.0;
			    }
		    }
		    return ret;
		}

		__device__ float divergence(float *y, int i, int j, int k, int l, int width, int height,
					int nK, int nL) {

   		    unsigned int strp_w = 2*height*nK*nL;
		    unsigned int strp_h = 2*nK*nL;
		    unsigned int strp_g = nK*nL;
		    unsigned int strp_k = nL;

	    	int idxx1 = strp_w*(i-1)+strp_h*j+strp_g*0+strp_k*k+l;
	    	int idxx2 = strp_w*i    +strp_h*j+strp_g*0+strp_k*k+l;
	    	int idxy1 = strp_w*i+strp_h*(j-1)+strp_g*1+strp_k*k+l;
	    	int idxy2 = strp_w*i+strp_h*j    +strp_g*1+strp_k*k+l;

		    float ret = 0;

		    if (i > 0) {
		    	ret = (y[idxx2] - y[idxx1]);
		    }
		    if (j > 0) {
		    	ret = ret + (y[idxy2] - y[idxy1]);
		    }
		    return ret;
		}

		//Simple merge sort implementation
		__device__ void merging(int low, int mid, int high, float *a, float *b, int thr)
		{
			int nK = {{ nK }};
			int l1, l2, i;
   			for(l1 = low, l2 = mid + 1, i = low; l1 <= mid && l2 <= high; i++) {
   				if(a[thr*nK + l1] <= a[thr*nK + l2])
   			    	b[thr*nK + i] = a[thr*nK + l1++];
   			    else
   			    	b[thr*nK + i] = a[thr*nK + l2++];
   			}
   			
   			while(l1 <= mid)    
   				b[thr*nK + i++] = a[thr*nK + l1++];
			
   			while(l2 <= high)   
   				b[thr*nK + i++] = a[thr*nK + l2++];
			
   			for(i = low; i <= high; i++)
   				a[thr*nK + i] = b[thr*nK + i];
		}
		
		__device__ void sort(int low, int high, float *a, float *b, int thr)
		{
			int mid;
			
			if(low < high) {
				mid = (low + high) / 2;
				sort(low, mid, a, b, thr);
				sort(mid+1, high, a, b, thr);
				merging(low, mid, high, a, b, thr);
			} else { 
				return;
			}   
		}

		__device__ void unitsimplex(float *ax, int nK, float *rx, int thr)
		{
			if (nK == 1) { rx[0] = 1; return; }
			float uv[{{nK}}];
			float vv[{{nK}}];
			float uvr[{{nK}}];
			float b[{{nK}}];
			float uvc = 0;
			int rho = 0;
			float uvs = 0;
			float lambda;
			float simpl;

			for (int k = 0; k < nK; k++) {
				uv[k] = ax[k];
			}
			sort(0, nK+1, uv, b, thr);
			for (int k = nK; k > 0; k--) {
				uvr[nK-k] = uv[k-1];
			}
			for (int k = 0; k < nK; k++) {
				uvc = uvc + uvr[k];
				vv[k] = uvr[k] + 1/float(k+1) - uvc/float(k+1);
				if (vv[k] > 0)
					rho = k;
			}
			for (int k = 0; k < rho+1; k++) {
				uvs = uvs + uvr[k];
			}
			lambda = (1 - uvs)/float(rho + 1);
			for (int k = 0; k < nK; k++) {
				simpl = ax[k] + lambda;
				if (simpl > 0)
					rx[k] = simpl;
				else
					rx[k] = 0;
			}
		}

		//def J1(x, h):
		//	return np.sum(np.linalg.norm(grad(x,h), axis = 2))


		//		self.cuda_glasso.prepared_call(grid_dimensions, block_dimensions,\
		//			z_gpu.gpudata, x_bar_gpu.gpudata, ps_norm_gpu.gpudata,\
		//			np.uint32(nElements), np.uint32(self.width), np.uint32(self.height),\
		//			np.uint32(self.nK), np.uint32(self.nL))

		__global__ void glasso(float *z, float *x_bar, float *output, int k,
							int nElements, int width, int height, int nK, int nL)
		{
		    __shared__ float partialSum[2*{{ block_size }}];
		    float sigma = {{ sigma }};

		    int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
		    unsigned int t = threadIdx.x;
		    unsigned int s = 2*blockIdx.x*blockDim.x;

		    unsigned int str_w = height*nK*nL;
		    unsigned int str_h = nK*nL;
		    unsigned int str_k = nL;
		    unsigned int str_l = 1;

	    	//Compute the pixel coordinates
	    	unsigned int i1 = (s+t)/height;
	    	unsigned int j1 = (s+t)-i1*height;
	    	unsigned int i2 = (s+blockDim.x+t)/height;
	    	unsigned int j2 = (s+blockDim.x+t)-i2*height;

	    	if ((s+t < nElements)) {
	    		//printf("Hi. This is thread %d, block %d \\n", t, blockIdx.x);
	    		//printf("Accessing pixel x1=%d, y1=%d \\n ", i1, j1);
	    		//printf("Accessing pixel x2=%d, y2=%d \\n ", i2, j2);
	    	}
	    	
	    	float summand;
	    	int idx; 

			//Compute Frobenius norm of (z + sigma*x_bar)_k
		    if ((s + t) < nElements)
		    {
		    	partialSum[t] = 0.0;
		    	for (unsigned int l = 0; l < nL; l++)
		    	{
		    		idx = str_w*i1+str_h*j1+str_k*k+l*str_l;
		    		summand = z[idx]+sigma*x_bar[idx];
		    		//printf("z at i = %d, j = %d, k = %d, l = %d is %f\\n", i1, j1, k, l, z[idx]);
		    		//printf("x_bar at i = %d, j = %d, k = %d, l = %d is %f\\n", i1, j1, k, l, x_bar[idx]);
			        partialSum[t] += summand*summand;
		    	}
		    }
		    else
		    {       
		        partialSum[t] = 0.0;
		    }
		    if ((s + blockDim.x + t) < nElements)
		    {   
	    		//printf("Writing pixel x2=%d, y2=%d \\n ", i2, j2);
		    	partialSum[blockDim.x + t] = 0.0;
		    	for (unsigned int idx = 0; idx < nL; idx++)
		    	{
		    		summand = z[str_w*i2+str_h*j2+str_k*k+str_l*idx]+sigma*x_bar[str_w*i2+str_h*j2+str_k*k+idx*str_l];
			        partialSum[blockDim.x + t] += summand*summand;
			    }
		    }
		    else
		    {
		        partialSum[blockDim.x + t] = 0.0;
		    }
		    //Traverse reduction tree
		    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2)
		    {
		    	__syncthreads();
		        if (t < stride)
		            partialSum[t] += partialSum[t + stride];
		    }
		    __syncthreads();
		    //Write the computed sum of the block to the output vector at correct index
		    if (t == 0 && (globalThreadId*2) < nElements)
		    {
		        output[blockIdx.x] = partialSum[t];
		    }
		}
		
		//self.cuda_unitball.prepared_call(grid_dimensions, block_dimensions,\
		//	x_bar_gpu.gpudata, y_gpu.gpudata, output_y_gpu.gpudata,\
		//	np.uint32(nElements), np.uint32(self.width), np.uint32(self.height),\
		//	np.uint32(self.nK), np.uint32(self.nL))

		__global__ void unitball(float *x_bar, float *y, float *y_out, int nElements,
			int width, int height, int nK, int nL)
			{

		    int globt = blockIdx.x*blockDim.x + threadIdx.x;

   		    unsigned int strp_w = 2*height*nK*nL;
		    unsigned int strp_h = 2*nK*nL;
		    unsigned int strp_g = nK*nL;
		    unsigned int strp_k = nL;

		    float sigma = {{ sigma }};

		    int idxp;
			float ay[2];
			float norm;
			float d;

		    if (globt < nElements) {
		    	//Compute the pixel coordinates
		    	unsigned int i = globt/height;
	    		unsigned int j = globt-i*height;

	    		for (int l = 0; l < nL; l++)
	    		{
	    			///////////////////////////////////////
					//y = unitball(y + sigma*grad(x_bar))//
	    			
	    			for (int k = 0; k < nK; k++) {
	    				for (int g = 0; g < 2; g++){
		    				idxp = strp_w*i+strp_h*j+strp_g*g+strp_k*k+l;
		    				ay[g] = y[idxp] + sigma*grad(x_bar,i,j,g,k,l,width,height,nK,nL);
	    				}
		    			norm = sqrt(ay[0]*ay[0] + ay[1]*ay[1]);
		    			if (2*norm > 1)
		    			{
		    				d = 2*norm;
		    			}
		    			else
		    			{
		    				d = 1;
		    			}
	    				for (int g = 0; g < 2; g++){
		    				idxp = strp_w*i+strp_h*j+strp_g*g+strp_k*k+l;
		    				//yold[g] = y[idxp];
		    				y_out[idxp] = ay[g]/d;
		    				//ynew[g] = y[idxp];
	    				}
					    if ((globt >= 0) && (k == 3) && (l == 3)) {
					    	//printf("(%d,%d) norm: %f, d: %f, yold1: %f, yold2: %f, ynew1: %f, ynew2: %f\\n", i, j, norm, d, yold[0], yold[1], ynew[0], ynew[1]);
					    	//printf("pixel i: %d, j: %d\\n", i, j);
					    }
	    			}
			    }
			}
		}

		//self.cuda_unitsimplex.prepared_call(grid_dimensions, block_dimensions,\
		//	x_gpu.gpudata, output_y_gpu.gpudata, z_gpu.gpudata, f_gpu.gpudata,\
		//	output_x_gpu.gpudata, np.uint32(nElements), np.uint32(self.width), np.uint32(self.height),\
		//	np.uint32(self.nK), np.uint32(self.nL))

		__global__ void project_unitsimplex(float *x, float *y, float *f,
			float *x_out, int nElements, int width, int height, int nK, int nL)
		{
		    int globt = blockIdx.x*blockDim.x + threadIdx.x;
		    int thr = threadIdx.x;

		    unsigned int str_w = height*nK*nL;
		    unsigned int str_h = nK*nL;
		    unsigned int str_k = nL;

		    float tau = {{ tau }};

		    int idx, kj;
		    float tmp;
			__shared__ float ax[{{nK}}*{{block_size}}];
			__shared__ float rx[{{nK}}*{{block_size}}];
			__shared__ float uv[{{nK}}*{{block_size}}];
			__shared__ float vv[{{nK}}*{{block_size}}];
			__shared__ float uvr[{{nK}}*{{block_size}}];
			
		    if (globt < nElements) {
		    	//Compute the pixel coordinates
		    	unsigned int i = globt/height;
	    		unsigned int j = globt-i*height;

	    		for (int l = 0; l < nL; l++)
	    		{
					/////////////////////////////////////////
					//x = unitsimplex(x - tau*(div(y)+z+f))//
					for (int k = 0; k < nK; k++) {
	    				idx = str_w*i+str_h*j+str_k*k+l;
						ax[thr*nK + k] = x[idx] - tau * (-divergence(y,i,j,k,l,width,height,nK,nL)+f[idx]);
					}

					//unitsimplex(ax, nK, rx, thr);
					
					//Unit simplex code is here instead...
					if (nK == 1)
					{
						rx[0] = 1;
					}
					else
					{
						float uvc = 0;
						int rho = 0;
						float uvs = 0;
						float lambda;
						float simpl;
						for (int k = 0; k < nK; k++) {
							uv[thr*nK + k] = ax[thr*nK + k];
						}

						//Just use insertion sort instead...
						//Pseudo-code
						//for i = 1 to length(A)
    					//	j = i
    					//	while j > 0 and A[j-1] > A[j]
    					//	    swap A[j] and A[j-1]
    					//	    j = j - 1
    					//	end while
						//end for
						for (int k = 0; k < nK; k++) {
							kj = k;
							while ((kj > 0) && (uv[thr*nK + kj-1] > uv[thr*nK + kj])) {
								//Swap 
								tmp = uv[thr*nK + kj];
								uv[thr*nK+kj] = uv[thr*nK+kj-1];
								uv[thr*nK+kj-1] = tmp;
								kj--;
							}
						}


						for (int k = nK; k > 0; k--) {
							uvr[thr*nK + nK-k] = uv[thr*nK + k-1];
						}
						for (int k = 0; k < nK; k++) {
							uvc = uvc + uvr[thr*nK + k];
							vv[thr*nK + k] = uvr[thr*nK + k] + 1/float(k+1) - uvc/float(k+1);
							if (vv[thr*nK + k] > 0)
								rho = k;
						}
						for (int k = 0; k < rho+1; k++) {
							uvs = uvs + uvr[thr*nK + k];
						}
						lambda = (1 - uvs)/float(rho + 1);
						for (int k = 0; k < nK; k++) {
							simpl = ax[thr*nK + k] + lambda;
							if (simpl > 0)
								rx[thr*nK + k] = simpl;
							else
								rx[thr*nK + k] = 0;
						}
					}	

					for (int k = 0; k < nK; k++) {
	    				idx = str_w*i+str_h*j+str_k*k+l;
	    				x_out[idx] = rx[thr*nK + k];
					}
			    }
			    if ((globt >= 0)) {
			    	//printf("(%d,%d) ax: %d, rx: %d\\n", i, j, ax, rx);
			    }
			}
		}

		//self.cuda_err.prepared_call(grid_dimensions, block_dimensions,\
		//	x_gpu.gpudata, x_old_gpu.gpudata, f_gpu.gpudata,\
		//	ps_err_gpu.gpudata, ps_ju_gpu.gpudata, ps_fu_gpu.gpudata,\
		//	np.uint32(nElements), np.uint32(self.width), np.uint32(self.height),\
		//	np.uint32(self.nK), np.uint32(self.nL))

		__global__ void error(float *x, float *x_old, float *f, float *ps_err,
					float *ps_ju, float *ps_fu, int nElements, int width, int height,
					int nK, int nL) {
			//err = np.linalg.norm(x-x_old)
			//ju = j_tv(x)
			//fu = np.sum(f*x)

		    __shared__ float partialSum_j[2*{{ block_size }}];
		    __shared__ float partialSum_f[2*{{ block_size }}];
		    __shared__ float partialSum_e[2*{{ block_size }}];

		    int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
		    unsigned int t = threadIdx.x;
		    unsigned int s = 2*blockIdx.x*blockDim.x;

		    unsigned int str_w = height*nK*nL;
		    unsigned int str_h = nK*nL;
		    unsigned int str_k = nL;

	    	//Compute the pixel coordinates
	    	unsigned int i1 = (s+t)/width;
	    	unsigned int j1 = (s+t)-i1*width;
	    	unsigned int i2 = (s+blockDim.x+t)/width;
	    	unsigned int j2 = (s+blockDim.x+t)-i2*width;

	    	int idx; 

	    	float sum_e, sum_f, sum_j;

		    if ((s + t) < nElements)
		    {
		    	for (unsigned int k = 0; k < nK; k++)
		    	{
			    	for (unsigned int l = 0; l < nL; l++)
			    	{
			    		idx = str_w*i1+str_h*j1+str_k*k+l;
			    		sum_e = x[idx]-x_old[idx];
			    		sum_j = x[idx]-x_old[idx];
			    		sum_f = f[idx]*x[idx];
				        partialSum_e[t] += sum_e*sum_e;
				        partialSum_j[t] += 0; // sum_j;
				        partialSum_f[t] += sum_f;
				    }
		    	}
		    }
		    else
		    {       
		        partialSum_e[t] = 0.0;
		        partialSum_j[t] = 0.0;
		        partialSum_f[t] = 0.0;
		    }
		    if ((s + blockDim.x + t) < nElements)
		    {   
		       	for (unsigned int k = 0; k < nK; k++)
			    {
			    	for (unsigned int l = 0; l < nL; l++)
			    	{
			       		idx = str_w*i2+str_h*j2+str_k*k+l;
			    		sum_e = x[idx]-x_old[idx];
			    		sum_j = x[idx]-x_old[idx];
			    		sum_f = f[idx]*x[idx];
				        partialSum_e[blockDim.x + t] += sum_e*sum_e;
				        partialSum_j[blockDim.x + t] += 0; // sum_j;
				        partialSum_f[blockDim.x + t] += sum_f;
				    }
				}
		    }
		    else
		    {
		        partialSum_e[blockDim.x + t] = 0.0;
		        partialSum_j[blockDim.x + t] = 0.0;
		        partialSum_f[blockDim.x + t] = 0.0;
		    }
		    //Traverse reduction tree
		    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2)
		    {
		    	__syncthreads();
		        if (t < stride)
		            partialSum_e[t] += partialSum_e[t + stride];
		            partialSum_f[t] += partialSum_f[t + stride];
		            partialSum_j[t] += partialSum_j[t + stride];
		    }
		    __syncthreads();
		    //Write the computed sum of the block to the output vector at correct index
		    if (t == 0 && (globalThreadId*2) < nElements)
		    {
		        ps_err[blockIdx.x] = partialSum_e[t];
		        ps_fu[blockIdx.x] = partialSum_f[t];
		        ps_ju[blockIdx.x] = partialSum_j[t];
		    }
		}

		}
		""")

		cuda_source = cuda_tpl.render(block_size=BLOCK_SIZE, tau = self.tau,\
					sigma = self.sigma, theta = self.theta, n_iter = self.n_iter,\
					eps = self.eps, nK = self.nK)
		cuda_module = SourceModule(cuda_source, no_extern_c=1)

		self.cuda_tpl = cuda_tpl 

		self.cuda_err = cuda_module.get_function("error")
		self.cuda_glasso = cuda_module.get_function("glasso")
		#self.cuda_seg = cuda_module.get_function("segmentation")
		self.cuda_err.prepare("PPPPPPiiiii")
		self.cuda_glasso.prepare("PPPiiiiii")
		#self.cuda_seg.prepare("PPPPPPiiiii")

		self.cuda_unitball = cuda_module.get_function("unitball")
		self.cuda_unitsimplex = cuda_module.get_function("project_unitsimplex")
		self.cuda_unitball.prepare("PPPiiiii")
		self.cuda_unitsimplex.prepare("PPPPiiiii")

		#self.cuda_test = cuda_module.get_function("hello")
		#self.cuda_test.prepare("PP")

	def run(self):
		nElements = self.width*self.height
		nBlocks = np.ceil(nElements/BLOCK_SIZE).astype(int)
		print 'No. elements:', nElements
		print 'No. blocks:', nBlocks
		grid_dimensions = (nBlocks, 1)
		block_dimensions = (BLOCK_SIZE, 1, 1)

		x_gpu = gpuarray.to_gpu(self.x)
		x_bar_gpu = gpuarray.to_gpu(self.x)
		x_old_gpu = gpuarray.to_gpu(self.x)
		y_gpu = gpuarray.to_gpu(self.y)
		f_gpu = gpuarray.to_gpu(self.f)

		x = self.x 
		x_old = self.x.copy()

		output_y = np.zeros(self.y.shape, dtype=np.float32)
		output_y_gpu = gpuarray.to_gpu(output_y)
		output_x = np.zeros(self.x.shape, dtype=np.float32)
		output_x_gpu = gpuarray.to_gpu(output_x)

		#The main loop runs on the CPU, since global synchronizations are difficult...
		#So for the moment this is the simplest way of implementing the algorithm
		print('====================================================================\nIter:\tdX:\t\tJ(u):\t\tf:\t\tPrimal objective:')
		for n in range(self.n_iter):
			######################################
			##Compute errors and such on the CPU##
			######################################

			err = np.linalg.norm(x-x_old)
			ju = self.J1(x, 1)
			fu = np.sum(self.f*x)
			obj = fu + ju

			print('%d\t%e\t%e\t%e\t%e'%(n, err, ju, fu, obj))
			if (err < self.eps) and (n > 0):
				break

			################################
			##Compute remainder of updates##
			################################

			x_old = x_gpu.get() 

			#y = unitball(y + sigma*grad(x_bar))
			self.cuda_unitball.prepared_call(grid_dimensions, block_dimensions,\
				x_bar_gpu.gpudata, y_gpu.gpudata, output_y_gpu.gpudata,\
				np.uint32(nElements), np.uint32(self.width), np.uint32(self.height),\
				np.uint32(self.nK), np.uint32(self.nL))
			cuda_driver.Context.synchronize()

			#x = unitsimplex(x - tau*(div(y)+z+f))
			self.cuda_unitsimplex.prepared_call(grid_dimensions, block_dimensions,\
				x_gpu.gpudata, output_y_gpu.gpudata, f_gpu.gpudata,\
				output_x_gpu.gpudata, np.uint32(nElements), np.uint32(self.width), np.uint32(self.height),\
				np.uint32(self.nK), np.uint32(self.nL))
			cuda_driver.Context.synchronize()

			x = output_x_gpu.get() 
			y = output_y_gpu.get()
			x_bar = x + self.theta*(x - x_old)

			#Reupload changes
			#x_gpu = output_x_gpu 
			#y_gpu = output_y_gpu 
			x_bar_gpu = gpuarray.to_gpu(x_bar) 
			x_gpu = gpuarray.to_gpu(x)
			y_gpu = gpuarray.to_gpu(y)

		u_s = x_gpu.get()

		return u_s

	def J1(self, x, h):
		return np.sum(np.linalg.norm(self.grad(x,h), axis = 2))

	def grad(self, u, h):
		k = u.shape[2]
		l = u.shape[3]
		p = np.zeros((u.shape[0], u.shape[1], 2, k, l))
		for j in range(l):
			for i in range(k):
				p[0:-1, :, 0, i, j] = (u[1:, :, i, j] - u[0:-1, :, i, j])/h
				p[:, 0:-1, 1, i, j] = (u[:, 1:, i, j] - u[:, 0:-1, i, j])/h
		return p 


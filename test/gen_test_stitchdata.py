import numpy as np 

from scipy.io import loadmat, savemat 
from matplotlib import pyplot as plt

#Load first video and find last frame's coordinates
nx = 30
ny = 30 

nF = 4
nV = 2 

#Generate u, v matrices for first video and second
u1 = np.zeros((nx, ny, nF))
v1 = np.zeros((nx, ny, nF))
u2 = np.zeros((nx, ny, nF))
v2 = np.zeros((nx, ny, nF))
u = np.zeros((nx, ny, 2*nF))
v = np.zeros((nx, ny, 2*nF))

#Fake info and param structures
params = []
info = []

#Image to translate
tx = 10 
ty = 10 
im = 128*np.ones(tx, ty)

#Generate u, v translation data for first video 
for idx in range(1,nF):
	u[im > 0,idx] = u[im > 0,idx-1] + vx1
	v[im > 0,idx] = v[im > 0,idx-1] + vy1
	u1[im > 0,idx] = u1[im > 0,idx-1] + vx1
	v1[im > 0,idx] = v1[im > 0,idx-1] + vy1

#Generate u, v translation data for second video
u[im > 0,nF] = u[im > 0,nF-1] + vx2
v[im > 0,nF] = v[im > 0,nF-1] + vy2

for idx in range(1,nF):
	ip = idx + nF
	u[im > 0,ip] = u[im > 0,ip-1] + vx2
	v[im > 0,ip] = v[im > 0,ip-1] + vy2
	u2[im > 0,idx] = u2[im > 0,idx-1] + vx2
	v2[im > 0,idx] = v2[im > 0,idx-1] + vy2

#Save
mdict = 

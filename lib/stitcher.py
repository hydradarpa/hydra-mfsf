###################################################
#gloo renderer stitching together MFSF video paths#
###################################################

#lansdell. October 12th 2016

import numpy as np 
from vispy import gloo
from vispy import app
import cv2 
from matplotlib import pyplot as plt

from cvtools import * 

VERT_SHADER = """
attribute vec3 a_position;
attribute vec2 a_texcoord;

void main (void) {
	gl_TexCoord[0] = vec4(a_texcoord.x, a_texcoord.y, 0.0, 0.0);
	gl_Position = vec4(a_position.x, a_position.y, a_position.z, 1.0);
}
"""

FRAG_SHADER = """
uniform sampler2D texture1;

void main()
{
	gl_FragColor = texture2D(texture1, gl_TexCoord[0].st);
}
"""

class Stitcher(app.Canvas):

	def __init__(self, u2, v2):

		title = 'The Stitcher'
		nx = u2.shape[1]
		ny = u2.shape[0]
		self.shape = (nx, ny)
		app.Canvas.__init__(self, title = title, show = False, size=(nx, ny), resizable=False)

		#Make triangles and vertex coords
		xv,yv = np.meshgrid(range(nx),range(ny))
		positions = np.hstack((xv.reshape((-1,1)),yv.reshape((-1,1))))

		nT = 2*(nx-1)*(ny-1)
		triangles = np.zeros((nT,3), dtype = np.uint32)
		count = 0
		for idxy in range(ny-1):
			for idxx in range(nx-1):
				p1 = idxy*nx + idxx 
				p2 = idxy*nx + idxx + 1 
				p3 = (idxy+1)*nx + idxx
				p4 = (idxy+1)*nx + idxx + 1
				t1 = [p1, p4, p2]
				t2 = [p1, p3, p4]
				triangles[count,:] = t1
				triangles[count + 1,:] = t2
				count += 2

		self.I = gloo.Texture2D(xv.astype(np.float32), format="luminance", internalformat="r32f")
		self.J = gloo.Texture2D(yv.astype(np.float32), format="luminance", internalformat="r32f")

		self.indices_buffer, self.vertex_data = self.loadMesh(positions, triangles, u2, v2)
		self._vbo = gloo.VertexBuffer(self.vertex_data)

		#Setup programs
		self._programi = gloo.Program(VERT_SHADER, FRAG_SHADER)
		self._programi['texture1'] = self.I
		self._programi.bind(self._vbo)
		self._programj = gloo.Program(VERT_SHADER, FRAG_SHADER)
		self._programj['texture1'] = self.J
		self._programj.bind(self._vbo)

		#Create FBOs, attach the color buffer and depth buffer
		self._rendertexi = gloo.Texture2D((self.shape + (1,)), format="luminance", internalformat="r32f")
		self._rendertexj = gloo.Texture2D((self.shape + (1,)), format="luminance", internalformat="r32f")

		self._fboi = gloo.FrameBuffer(self._rendertexi, gloo.RenderBuffer(self.shape))
		self._fboj = gloo.FrameBuffer(self._rendertexj, gloo.RenderBuffer(self.shape))

		#self.show()
		gloo.set_clear_color('black')
		gloo.set_viewport(0, 0, *self.physical_size)
		#gloo.set_viewport(0, 0, ny, nx)

		self.u2 = u2 
		self.v2 = v2 

	def run(self, u1, v1):
		#Render this baby
		(pi, pj) = self.render()
		#print 'pi.shape:', pi.shape 
		#print 'pj.shape:', pj.shape 

		#Make a plot of the babies to see what they look like...
		#f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
		#ii = ax1.imshow(pi)
		#ij = ax2.imshow(pj)
		#ax1.set_xlim((0, self.shape[0]))
		#ax1.set_ylim((0, self.shape[1]))
		#ax2.set_xlim((0, self.shape[0]))
		#ax2.set_ylim((0, self.shape[1]))
		#f.colorbar(ii)
		#plt.show()

		uf = np.squeeze(u1[:,:,-1])
		vf = np.squeeze(v1[:,:,-1])

		u = np.zeros(self.u2.shape)
		v = np.zeros(self.v2.shape)

		(nx, ny) = self.shape
		for i in range(ny):
			print("Shifting row %d"%i)
			for j in range(nx):
				#Get a path's final location in first video
				ip = max(0, min(ny-1, int(i + vf[i,j])))
				jp = max(0, min(nx-1, int(j + uf[i,j])))
				#Find that location in the second video's reference frame
				fix = int(pi[ip, jp])
				fiy = int(pj[ip, jp])
				#print 'ip, jp, fiy, fix'
				#print ip, jp, fiy, fix
				#Use that reference pixel's optic flow data, with reference point adjusted
				u[i,j,:] = fix - j + self.u2[fiy,fix,:]
				v[i,j,:] = fiy - i + self.v2[fiy,fix,:]

		return u, v

	def on_resize(self, event):
		pass

	def draw(self, event):
		#gloo.set_state('additive')
		#gloo.set_state(cull_face = False)
		#gloo.set_cull_face('front')
		gloo.clear()
		self._program_flow.bind(self._vbo)
		self._programi.draw('triangles', self.indices_buffer)

	def render(self):
		with self._fboi:
			gloo.clear()
			self._programi.draw('triangles', self.indices_buffer)
			pixelsi = gloo.read_pixels(out_type = np.float32)[:,:,0]
		with self._fboj:
			gloo.clear()
			self._programj.draw('triangles', self.indices_buffer)
			pixelsj = gloo.read_pixels(out_type = np.float32)[:,:,0]
		return (pixelsi, pixelsj)

	#Load mesh data
	def loadMesh(self, vertices, triangles, u, v):
		# Create vertices and texture coords, combined in one array for high performance
		self.nP = vertices.shape[0]
		self.nT = triangles.shape[0]

		vertex_data = np.zeros(self.nP, dtype=[('a_position', np.float32, 3),
			('a_texcoord', np.float32, 2)])

		warped_vertices = np.zeros(vertices.shape)
		for i,p in enumerate(vertices):
			#wi = [p[0] + u[p[0], p[1], 0], p[1] + v[p[0], p[1], 0]]
			wi = [p[0] + u[p[1], p[0], 0], p[1] + v[p[1], p[0], 0]]
			warped_vertices[i,:] = wi

		verdata = np.zeros((self.nP,3))
		uvdata = np.zeros((self.nP,2))

		#rescale
		(nx,ny) = self.shape
		verdata[:,0] = 2*warped_vertices[:,0]/nx-1
		verdata[:,1] = 2*warped_vertices[:,1]/ny-1
		#verdata[:,0] = warped_vertices[:,0]/nx
		#verdata[:,1] = warped_vertices[:,1]/ny

		#Does this need to be flipped?
		verdata[:,1] = -verdata[:,1]
		#verdata[:,0] = -verdata[:,0]

		uvdata = vertices.astype(np.float32)
		uvdata[:,0] = uvdata[:,0]/nx
		uvdata[:,1] = uvdata[:,1]/ny

		vertex_data['a_position'] = verdata
		vertex_data['a_texcoord'] = uvdata 

		indices = triangles.reshape((1,-1))
		indices_buffer = gloo.IndexBuffer(indices)
	
		return indices_buffer, vertex_data
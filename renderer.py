#################################
#gloo renderer for Kalman filter#
#################################

#lansdell. Feb 11th 2016

import numpy as np 
from vispy import gloo
from vispy import app
from imgproc import findObjectThreshold
import cv2 

from cuda import CUDAGL 

VERT_SHADER = """ // simple vertex shader

attribute vec3 a_position;
attribute vec3 a_velocity;
attribute vec2 a_texcoord;
uniform vec4 u_color;
varying vec4 v_color;

void main (void) {
	v_color = u_color;
	// Pass tex coords
	gl_TexCoord[0] = vec4(a_texcoord.x, a_texcoord.y, 0.0, 0.0);
	// Calculate position
	gl_Position = vec4(a_position.x, a_position.y, a_position.z, 1.0);
}
"""

FRAG_SHADER = """ // simple fragment shader
uniform sampler2D texture1;

void main()
{
	gl_FragColor = texture2D(texture1, gl_TexCoord[0].st);
}
"""

FRAG_SHADER_RED = """ // simple fragment shader
uniform sampler2D texture1;

void main()
{
	vec4 tex1 = texture2D(texture1, gl_TexCoord[0].st);
	gl_FragColor = vec4(tex1.r, 0, 0, tex1.a);
}
"""

FRAG_SHADER_GREEN = """ // simple fragment shader
uniform sampler2D texture1;

void main()
{
	vec4 tex1 = texture2D(texture1, gl_TexCoord[0].st);
	gl_FragColor = vec4(0, tex1.g, 0, tex1.a);
}
"""

FRAG_SHADER_FLOWX = """ // simple fragment shader
varying vec4 v_color;

void main()
{
	gl_FragColor = v_color;
}
"""

FRAG_SHADER_FLOWY = """ // simple fragment shader
varying vec4 v_color;

void main()
{
	gl_FragColor = v_color;
}
"""

FRAG_SHADER_FLOW = """ // simple fragment shader
varying vec4 v_color;

void main()
{
	gl_FragColor = v_color;
}
"""

FRAG_SHADER_LINES = """ // simple fragment shader
varying vec4 v_color;

void main()
{
	gl_FragColor = v_color;
}
"""

class Renderer(app.Canvas):

	def __init__(self, distmesh, vel, nx, im1, eps_R, cuda):
		self.state = 'texture'
		title = 'Hydra tracker. Displaying %s state (space to toggle)' % self.state
		app.Canvas.__init__(self, keys='interactive', title = title)
		self.size = (nx, nx)
		self.indices_buffer, self.outline_buffer, self.vertex_data, self.quad_data, self.quad_buffer = self.loadMesh(distmesh.p, vel, distmesh.t, nx)
		self._vbo = gloo.VertexBuffer(self.vertex_data)
		self._quad = gloo.VertexBuffer(self.quad_data)
		self.current_frame = im1 
		self.current_texture = gloo.Texture2D(im1)
		self.init_texture = gloo.Texture2D(im1)

		#Setup programs
		self._program = gloo.Program(VERT_SHADER, FRAG_SHADER)
		self._program['texture1'] = self.init_texture
		self._program.bind(self._vbo)
		self._program_lines = gloo.Program(VERT_SHADER, FRAG_SHADER_LINES)
		self._program_lines['u_color'] = 0, 1, 1, 1
		self._program_lines.bind(self._vbo)
		self._program_flowx = gloo.Program(VERT_SHADER, FRAG_SHADER_FLOWX)
		self._program_flowx['u_color'] = 1, 0, 0, 1
		self._program_flowx.bind(self._vbo)
		self._program_flowy = gloo.Program(VERT_SHADER, FRAG_SHADER_FLOWY)
		self._program_flowy['u_color'] = 1, 0, 0, 1
		self._program_flowy.bind(self._vbo)
		self._program_flow = gloo.Program(VERT_SHADER, FRAG_SHADER_FLOW)
		self._program_flow['u_color'] = 0, 1, 0, 1
		self._program_flow.bind(self._vbo)
		self._program_red = gloo.Program(VERT_SHADER, FRAG_SHADER_RED)
		self._program_green = gloo.Program(VERT_SHADER, FRAG_SHADER_GREEN)

		#Create FBOs, attach the color buffer and depth buffer
		self.shape = (nx, nx)
		self._rendertex1 = gloo.Texture2D((self.shape + (3,)))
		self._rendertex2 = gloo.Texture2D((self.shape + (3,)))
		self._rendertex3 = gloo.Texture2D((self.shape + (3,)))
		self._fbo1 = gloo.FrameBuffer(self._rendertex1, gloo.RenderBuffer(self.shape))
		self._fbo2 = gloo.FrameBuffer(self._rendertex2, gloo.RenderBuffer(self.shape))
		self._fbo3 = gloo.FrameBuffer(self._rendertex3, gloo.RenderBuffer(self.shape))

		#import rpdb2 
		#rpdb2.start_embedded_debugger("asdf")
		gloo.set_viewport(0, 0, nx, nx)
		gloo.set_clear_color('black')
		self._timer = app.Timer('auto', connect=self.update, start=True)
		self.show()

		self.cudagl = CUDAGL(self._rendertex1, eps_R, self._fbo1)

	def on_resize(self, event):
		width, height = event.physical_size
		#gloo.set_viewport(0, 0, width, height)

	def on_draw(self, event):
		#Render the current positions to FBO1 
		gloo.clear()
		with self._fbo1:
			self._program.draw('triangles', self.indices_buffer)
		#Render the current velocities to FBO2
		gloo.clear()
		with self._fbo2:
			self._program_flowx.draw('triangles', self.indices_buffer)
		gloo.clear()
		with self._fbo3:
			self._program_flowy.draw('triangles', self.indices_buffer)

		#Turn on additive blending
		gloo.set_state('additive')
		gloo.clear()
		#Summary render to main screen
		if self.state == 'texture':
			self._program.draw('triangles', self.indices_buffer)
		else:
			self._program_red['texture1'] = self.current_texture
			self._program_red.bind(self._quad)
			self._program_red.draw('triangles', self.quad_buffer)
			self._program_green.bind(self._vbo)
			self._program_green['texture1'] = self.init_texture
			self._program_green.draw('triangles', self.indices_buffer)
		#Draw wireframe, too
		self._program_lines.draw('lines', self.outline_buffer)

	def on_key_press(self, event):
		if event.key in [' ']:
			if self.state == 'texture':
				self.state = 'overlay'
			else:
				self.state = 'texture'
			self.title = 'Hydra tracker. Displaying %s state (space to toggle)' % self.state
			self.update()
		#if event.key in ['q', 'Q']:
		#	app.quit()

	def update_vertex_buffer(self, vertices, velocities):
		verdata = np.zeros((self.nP,3))
		veldata = np.zeros((self.nP,3))
		#rescale
		verdata[:,0:2] = 2*vertices/self.nx-1
		verdata[:,1] = -verdata[:,1]
		veldata[:,0:2] = 2*velocities/self.nx-1
		veldata[:,1] = -veldata[:,1]
		self.vertex_data['a_position'] = verdata
		self.vertex_data['a_velocity'] = veldata 
		self._vbo = gloo.VertexBuffer(self.vertex_data)
		self._program.bind(self._vbo)
		self._program_lines.bind(self._vbo)
		self._program_flowx.bind(self._vbo)
		self._program_flowy.bind(self._vbo)

	#Load mesh data
	def loadMesh(self, vertices, velocities, triangles, nx):
		# Create vetices and texture coords, combined in one array for high performance
		self.nP = vertices.shape[0]
		self.nT = triangles.shape[0]
		self.nx = nx
		vertex_data = np.zeros(self.nP, dtype=[('a_position', np.float32, 3),
			('a_texcoord', np.float32, 2), ('a_velocity', np.float32, 3)])
		verdata = np.zeros((self.nP,3))
		veldata = np.zeros((self.nP,3))
		uvdata = np.zeros((self.nP,2))
		outlinedata = np.zeros((self.nT, 6))
		#rescale
		verdata[:,0:2] = 2*vertices/nx-1
		verdata[:,1] = -verdata[:,1]
		veldata[:,0:2] = 2*velocities/nx-1
		veldata[:,1] = -veldata[:,1]
		uvdata = vertices/nx
		vertex_data['a_position'] = verdata
		vertex_data['a_texcoord'] = uvdata 
		vertex_data['a_velocity'] = veldata 
		indices = triangles.reshape((1,-1)).astype(np.uint16)
		indices_buffer = gloo.IndexBuffer(indices)
	
		for idx, t in enumerate(triangles):
			outlinedata[idx,0] = t[0]
			outlinedata[idx,1] = t[1]
			outlinedata[idx,2] = t[1]
			outlinedata[idx,3] = t[2]
			outlinedata[idx,4] = t[2]
			outlinedata[idx,5] = t[0]
		outline = outlinedata.reshape((1,-1)).astype(np.uint16)
		outline_buffer = gloo.IndexBuffer(outline)

		quad_data = np.zeros(4, dtype=[('a_position', np.float32, 3),
				('a_texcoord', np.float32, 2), ('a_velocity', np.float32, 3)])
		quad_ver = np.array([[-1.0, 1.0, 0.0],  [-1.0, -1.0, 0.0],
							 [ 1.0, 1.0, 0.0], [ 1.0, -1.0, 0.0,] ], np.float32)
		quad_coord = np.array([  [0.0, 0.0], [0.0, 1.0],
								[1.0, 0.0], [1.0, 1.0] ], np.float32)
		quad_vel = np.zeros((4,3))
		quad_data['a_texcoord'] = quad_coord 
		quad_data['a_position'] = quad_ver 
		quad_data['a_velocity'] = quad_vel 
		quad_triangles = np.array([[0, 1, 2], [1, 2, 3]])
		quad_indices = quad_triangles.reshape((1,-1)).astype(np.uint16)
		quad_buffer = gloo.IndexBuffer(quad_indices)

		return indices_buffer, outline_buffer, vertex_data, quad_data, quad_buffer

	def update_frame(self, y_im):
		self.current_frame = y_im 
		self.current_texture = gloo.Texture2D(y_im)

	def z(self, y_im):
		#return self.cudagl.z(y_im)
		return self.cudagl.z_CPU(y_im)


class VideoStream:
	def __init__(self, fn, threshold):
		print("Creating video stream from " + fn)
		print("Background subtraction below intensity " + str(threshold))
		self.threshold = threshold
		self.cap = cv2.VideoCapture(fn)
		ret, frame = self.cap.read()
		assert ret, "Cannot open %s" % fn 
		nx,ny = frame.shape[0:2]
		self.nx = nx 
		self.ny = ny
		self.frame = frame
		self.frame_orig = frame.copy()
		self.grayframe = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)

	def read(self, backsub = True):
		ret, frame = self.cap.read()
		self.frame = frame 
		self.grayframe = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
		if not backsub:
			return ret, self.frame, self.grayframe
		else:
			(mask, ctrs, fd) = findObjectThreshold(self.frame, threshold = self.threshold)
			#Apply mask
			backframe = np.multiply(np.dstack((mask, mask, mask)), self.frame)
			backgrayframe = np.multiply(mask, self.grayframe)		
			return ret, backframe, backgrayframe 

	def current_frame(self, backsub = True):
		if backsub:
			(mask, ctrs, fd) = findObjectThreshold(self.frame, threshold = self.threshold)
			frame = np.multiply(np.dstack((mask, mask, mask)), self.frame)
		else:
			frame = self.frame 
		return frame 

	def gray_frame(self, backsub = True):
		if backsub:
			(mask, ctrs, fd) = findObjectThreshold(self.frame, threshold = self.threshold)
			frame = np.multiply(mask, self.grayframe)
		else:
			frame = self.grayframe 
		return frame 

	def backsub(self, im = None):
		(mask, ctrs, fd) = findObjectThreshold(self.frame, threshold = self.threshold)
		if im is None:
			return mask, ctrs, fd
		else:
			if len(im.shape) == 2:
				return np.multiply(mask, im)
			#Probably a better way to do this...
			elif im.shape[2] == 2:
				return np.multiply(np.dstack((mask, mask)), im)
			elif im.shape[2] == 3:
				return np.multiply(np.dstack((mask, mask, mask)), im)

	def isOpened(self):
		return self.cap.isOpened()
#####################################################
#gloo renderer for MFSF dewarping of a set of points#
#####################################################

#lansdell. May 25th 2017
from stitcher import * 

class Warper(Stitcher):
	def update(self, u2, v2):
		self.indices_buffer, self.vertex_data = self.loadMesh(self.positions, self.triangles, u2, v2)
		self._vbo = gloo.VertexBuffer(self.vertex_data)
		self._programi.bind(self._vbo)
		self._programj.bind(self._vbo)

	def run(self, pts):
		(pi, pj) = self.render()
		warp_pts = np.zeros(pts.shape)
		for idx in range(len(pts)):
			i = pts[idx,0]
			j = pts[idx,1]
			fix = int(pi[i, j])
			fiy = int(pj[i, j])
			warp_pts[idx,0] = fix
			warp_pts[idx,1] = fiy
		return warp_pts
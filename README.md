# HydraMFSF
Python code for Hydra optical flow, behavior and neural analysis.

Analysis of MFSF (multi-frame subspace constrained optic flow -- http://www0.cs.ucl.ac.uk/staff/lagapito/subspace_flow/) tracking of Hydra body/neurons.

lansdell 2016

Dependencies:
* Vispy*
* Numpy
* PyCuda**
* DistMesh 
* OpenCV2
* cvtools (https://github.com/benlansdell/cvtools)
* matplotlib

Notes:
* *Uses OpenGL rendering. If using remotely, you'll need to set up a VirtualGL server
* **If have a CUDA compatible graphics card

![alt tag](https://github.com/benlansdell/hydra-mfsf/blob/master/hydra_tracked.png)

# References
[1] Ravi Garg, Anastasios Roussos, Lourdes Agapito, "A Variational Approach to Video Registration with Subspace Constraints", International journal of computer vision 104 (3), 286-314, 2013 

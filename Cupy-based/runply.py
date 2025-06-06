from time import time
import argparse
import os
import numpy as np
import trimesh
import os

N=5000
file_name='./data/ply/light.ply'
file_name2=os.path.basename(file_name)
mesh = trimesh.load_mesh(file_name)
points,faces=trimesh.sample.sample_surface_even(mesh,N)
mu, sigma = 0, 1 
s = np.random.normal(mu, sigma, points.shape) *0.000
normal=mesh.face_normals[faces]
PointCloud=np.array(points)+s
result='./result'
np.savetxt(result+'/'+file_name2[:-4]+'.xyz',  np.c_[PointCloud],fmt='%f',delimiter=' ')
sample=result+'/'+file_name2[:-4]+'.xyz'
A=r"python runAGR.py   --alpha 2.0 -wk 16 "
recon_cmdAGR= f"{A} {sample}"
os.system(recon_cmdAGR)
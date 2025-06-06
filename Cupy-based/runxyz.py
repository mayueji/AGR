from time import time
import argparse
import os
import numpy as np
import trimesh
import os

file_name='Utah_teapot_3000.xyz'
path=r'./data/xyz'
sample=path+'/'+file_name
A=r"python runAGR.py   --alpha 2 -wk 16 -wmin 0.04 "
recon_cmdAGR= f"{A} {sample}"
os.system(recon_cmdAGR)

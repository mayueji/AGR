from time import time
import argparse
import os
import numpy as np
import trimesh
import os

path =r'./data'.replace("\\",'/') 
currentpath = os.getcwd().replace("\\",'/') 
result= os.path.join(currentpath,'result').replace("\\",'/')
isExists = os.path.exists(result) 
if not isExists: 
    os.mkdir(result) 
file_name='Armadillo_40000.xyz'
sample=path+'/'+file_name
A=r"python main_AGR.py"
recon_cmdAGR= f"{A} {sample}"+" --width_config l0 --tqdm "
os.system(recon_cmdAGR)
import os
run2_EXE = r"./main_GaussReconCUDA" #OAGR 
# run2_EXE = r"./main_GaussReconCUDA0" #GR 

path1=r'./results'
path2=r'./ply'

for file_name in os.listdir(path1):
    if file_name[-8:-4]=='true':
        continue
    file1=path1+"/"+ file_name
    file2=path2+'/'+ file_name[:-4]+'.ply'
    I=r' -i '+file1
    II=r' -o '+file2
    recon_cmd3 = f"{run2_EXE} "+I +II 
    print(recon_cmd3)
    os.system(recon_cmd3)
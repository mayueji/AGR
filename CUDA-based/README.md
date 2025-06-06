# AGR
Based on WNNC(2024 SIGGRAPH Asia), we provide the latest implementations for [Gauss surface reconstruction](GR). We also provide the anisotropic surface reconstruction for oriented points (OAGR) which is faster than [PoissonRecon] and has better performance than GR on thin structures.

### Usage
![Special note: We only provide GPU version code]
![Special note: We only provide GPU version code]
![Special note: We only provide GPU version code]
1. For AGR:
```bash
cd ext
pip install -e .
cd ..

# width is important for normal quality, we provide a few presets through --width_config

# for clean uniform samples, use l0
python main_AGR.py data/Armadillo_40000.xyz --width_config l0 --tqdm

# for noisy or non-uniform points, use configs l1 (small noise) ~ l5 (large noise) depending on the noise level
# a higher level gives smoother normals and better resilience to noise
python main_AGR.py data/bunny_noised.xyz --width_config l1 --tqdm
...
python main_AGR.py data/bunny_noised.xyz --width_config l5 --tqdm

# the user can also use custom widths:
python main_AGR.py data/bunny_noised.xyz --width_config custom --wsmin 0.03 --wsmax 0.12 --tqdm

# to see a complete list of options:
python main_main_AGR.py -h
```

[Input] If the input is mesh(.ply), the user can change the path ans settings in AGR-ply.py and run:
```
python AGR-ply.py
```

[Input] If the input is point(.xyz), the user can change the path ans settings in AGR-xyz.py and run:
```
python AGR-xyz.py
```


2. For Gauss surface reconstruction:
You may  download [ANN 1.1.2](https://www.cs.umd.edu/~mount/ANN/) and unpack to `ext/gaussrecon_src/ANN`.(In fact, compared to WNNC, we have already downloaded and configured usrs. If usrs encounters problems while running WNNC, You can run this code environment, which is compatible with WNNC) Run `make` there. Then go back to the main repository directory, and:
```bash 
sh build_GR_cpu.sh
sh build_GR_cuda.sh

./main_GaussReconCPU -i <input.xyz> -o <output.ply>
./main_GaussReconCUDA -i <input.xyz> -o <output.ply>
```

For convenience, we provide a Python file for batch reconstruction through AGR.
python reconstruction.py





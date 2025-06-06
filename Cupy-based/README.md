# AGR   

### Environment and Dependencies

This program has been tested on Ubuntu 20.04 with Python 3.8 and CUDA 11.1.  This program requires these Python packages: `numpy`, `cupy`, `scipy` and `tqdm`.

### Compiling Source Files

This program uses third-party libraries [CLI11](https://github.com/CLIUtils/CLI11) and [cnpy](https://github.com/rogersce/cnpy). We provide copies of them in this repository for convenicence. If you have [cmake](https://cmake.org/), you can build with:

```bash
mkdir build
cd build
cmake ..
make -j8
```

Or you can also build with `g++` directly:

```bash
cd src
g++ PGRExportQuery.cpp Cube.cpp Geometry.cpp MarchingCubes.cpp Mesh.cpp Octnode.cpp Octree.cpp ply.cpp plyfile.cpp cnpy/cnpy.cpp -ICLI11 -o ../apps/PGRExportQuery -lz -O2
g++ PGRLoadQuery.cpp Cube.cpp Geometry.cpp MarchingCubes.cpp Mesh.cpp Octnode.cpp Octree.cpp ply.cpp plyfile.cpp cnpy/cnpy.cpp -ICLI11 -o ../apps/PGRLoadQuery -lz -O2
```

If successful, this will generate two executables `PGRExportQuery` and `PGRLoadQuery` in `ParametricGaussRecon/apps`. The former builds an octree and exports grid corner points for query; the latter loads query values solved by PGR and performs iso-surfacing.

(For ease of users, we have compiled and can directly use the two files mentioned above (Linux system). Please grant them executable permissions.)

### All code must run on a GPU, only the GPU version is provided here.(It can be run directly on a Linux server (ensuring that PGRExportQuery and PGRLoadQuery have set read-write permissions))

```bash
python runAGR.py data/xyz/bunny_10000.xyz --alpha 2 -wk 16	

This will create a folder `results/bunny_10000.xyz` together with three subfolders: `recon`, `samples` and `solve` :

- `recon` contains the reconstructed meshes in PLY format.
- `samples` contains the input point cloud in XYZ format, the normalized input point cloud and the octree grid corners as query set in NPY format.
- `solve` contains the solved Linearized Surface Elements in XYZ and NPY formats, the queried values, query set widths in NPY format, and the iso-value in TXT format.
```

```bash
python runAGR.py data/xyz/Utah_teapot_3000.xyz --alpha 2 -wk 16	
```

For the convenience of operation, we provide a version that can be directly reconstructed from. py (randomly selected points from the mesh), with a default selection of 5K points in this version.

python runply.py data/ply/bunny.ply --alpha 2 -wk 16
python runply.py data/ply/light.ply 

More files(.ply) can be obtained by run:

python download_datasets_abc.py
python download_datasets_thingi10k.py
python download_datasets_real_world.py
python download_datasets_famous.py
in /data


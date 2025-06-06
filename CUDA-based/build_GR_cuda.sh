nvcc -O3 \
    ext/gaussrecon_src/Cube.cpp \
    ext/gaussrecon_src/MarchingCubes.cpp \
    ext/gaussrecon_src/Octree.cpp \
    ext/gaussrecon_src/plyfile.cpp \
    ext/gaussrecon_src/Geometry.cpp \
    ext/gaussrecon_src/Mesh.cpp \
    ext/gaussrecon_src/main_GaussRecon_cuda.cu \
    ext/gaussrecon_src/ply.cpp \
    ext/gaussrecon_src/ReconOctNode.cpp \
    ext/gaussrecon_src/ANNAdapter.cpp \
    ext/wn_treecode/wn_treecode_cuda/wn_treecode_cuda_kernels.cu \
    ext/wn_treecode/wn_treecode_cpu/wn_treecode_cpu_treeutils.cpp \
    -I ext/wn_treecode/wn_treecode_cpu/ \
    -I ext/wn_treecode/wn_treecode_cuda/ \
    -I ext/gaussrecon_src/CLI11 -Iext/gaussrecon_src/ANN/include \
    -L ext/gaussrecon_src/ANN/lib \
    -lz -lANN \
    -o main_GaussReconCUDA 
# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.8

# compile CUDA with /usr/local/cuda/bin/nvcc
CUDA_FLAGS =  -gencode=arch=compute_75,code=\"sm_75,compute_75\" -rdc=true -DWMMA  -Xcompiler -Wall --expt-extended-lambda --expt-relaxed-constexpr --std=c++11 -Xcompiler -O3 -O3 -DNDEBUG  

CUDA_DEFINES = 

CUDA_INCLUDES = -I/workspace/ix/FasterTransformer-master_2 -I/usr/local/cuda/include -I/root/miniconda2/lib/python2.7/site-packages/tensorflow/include 

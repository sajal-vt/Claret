# Claret : A Portable Parallel MDS Tool
Claret is our fast and portable parallel WMDS tool that combines algorithmic concepts adapted and extended from the stochastic force-based MDS (SF-MDS) and Glimmer. To further improve Claret's performance for real-time data analysis, we propose a preprocessing step that computes approximate weighted Euclidean distances by combining a novel data mapping called stretching and Johnson Lindestrauss' lemma in O(log d) time in place of the original O(d) time. This preprocessing step reduces the complexity of WMDS from O(f(n)d) to O(f(n) log d), which for large d is a significant computational gain.

## Requirement
1. OpenCL for CPU ([Installation Instruction](https://software.intel.com/en-us/articles/opencl-drivers))
2. OpenCL for GPU ([Installation Instruction for NVIDIA GPUs](https://developer.nvidia.com/opencl))
3. For running with Python ([Install PyOpenCL](https://mathema.tician.de/software/pyopencl/))

## Building MDS/WMDS

1. For C++ MDS
```
g++ -L /usr/local/cuda-9.0/lib64/ ../source/host/cpp/MDS.cpp -lOpenCL -o mds
```
2. For C++ WMDS
```
g++ -L /usr/local/cuda-9.0/lib64/ ../source/host/cpp/WMDS.cpp -lOpenCL -o wmds
```
3. For Python MDS/WMDS
```
No separate building is required
```

## Using Python distributable package named claret
We have included a python distributable package named claret in the dist directory of the project. We will add a C++ distributable in a future commit.
To use wmds from claret in your program:
1. Put dist/claret folder in the same directory as your program
2. Import wmds from claret
```
from claret import wmds
```
3. Call reduce method from wmds. Pass 2D data array, weigths and 2 as the projected dimension
```
wmds.reduce(data, weights, 2)
```

#!/usr/bin/env bash

nvcc --compiler-options '-fPIC' -o gpu/libgmat.so --shared gpu/cuda.cu

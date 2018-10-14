#!/usr/bin/env bash

nvcc -Wreorder --compiler-options '-fPIC' -o gpu/libgmat.so --shared gpu/cuda.cu

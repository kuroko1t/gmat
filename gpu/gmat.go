// +build gpu

// Copyright 2018 kurosawa. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

package gpu

// #cgo CFLAGS: -I/usr/local/cuda/targets/x86_64-linux/include/
// #cgo LDFLAGS: -L/usr/local/cuda/lib64/ -L/usr/lib/x86_64-linux-gnu -lcudart -lcuda -lcudnn -lcublas
// #include </usr/local/cuda/include/cuda_runtime.h>
// #include "cublas_v2.h"
// #include </usr/include/cudnn.h>
import "C"
import "unsafe"
//import "fmt"

type Handle struct {
	cublasHandle C.cublasHandle_t
}

type Tensor struct {
	CPU   [][]float64
	CPU4D [][][][]float64
	GPU   *C.float
	//GPU   []float32
	Shape []int
}

func (handle *Handle) Malloc(n int) *C.float {
	var z *C.float
	gpuptr := unsafe.Pointer(z)
	typeSize := int(unsafe.Sizeof(float32(0)))
	gpuSize := C.size_t(n * typeSize)
	C.cudaMalloc(&gpuptr, gpuSize)
	return (*C.float)(gpuptr)
}

func (handle *Handle) CopyH2D(x [][]float64) (int, int, *C.float) {
	n := len(x)
	m := len(x[0])
	var z *C.float
	gpuptr := unsafe.Pointer(z)
	typeSize := int(unsafe.Sizeof(float32(0)))
	gpuSize := C.size_t(m * n * typeSize)
	C.cudaMalloc(&gpuptr, gpuSize)
	x32 := d2f(x)
	C.cudaMemcpy(gpuptr, unsafe.Pointer(&x32[0]), gpuSize, C.cudaMemcpyHostToDevice)
	return n, m, (*C.float)(gpuptr)
}

func (handle *Handle) CopyD2H(shape []int, gpuptr *C.float) [][]float64 {
	n := shape[0]
	m := shape[1]
	z := make([]float32, n*m)
	typeSize := int(unsafe.Sizeof(float32(0)))
	cudaCheck(C.cudaMemcpy(unsafe.Pointer(&z[0]),
		unsafe.Pointer(gpuptr),
		C.size_t(n*m*typeSize), C.cudaMemcpyDeviceToHost))
	z64 := f2d(z, n, m)
	return z64
}

func (handle *Handle) Dot(x, y *C.float, m, n, k int) *C.float {
	z := handle.Malloc(m * n)
	if handle.cublasHandle == nil {
		handle.cublasHandle = cublaInit()
	}
	var alpha C.float = 1
	var beta C.float = 1
	cublasCheck(C.cublasSgemm(handle.cublasHandle,
		C.CUBLAS_OP_N, C.CUBLAS_OP_N,
		C.int(m), C.int(n), C.int(k),
		&alpha,
		x, C.int(m),
		y, C.int(k),
		&beta,
		z, C.int(m)))
	return z
}

func (handle *Handle) TDot(x, y *C.float, m, n, k int) *C.float {
	z := handle.Malloc(m * n)
	if handle.cublasHandle == nil {
		handle.cublasHandle = cublaInit()
	}
	var alpha C.float = 1
	var beta C.float = 1
	cublasCheck(C.cublasSgemm(handle.cublasHandle,
		C.CUBLAS_OP_T, C.CUBLAS_OP_N,
		C.int(m), C.int(n), C.int(k),
		&alpha,
		x, C.int(k),
		y, C.int(k),
		&beta,
		z, C.int(m)))
	return z
}

func (handle *Handle) DotT(x, y *C.float, m, n, k int) *C.float {
	z := handle.Malloc(m * n)
	if handle.cublasHandle == nil {
		handle.cublasHandle = cublaInit()
	}
	var alpha C.float = 1
	var beta C.float = 1
	cublasCheck(C.cublasSgemm(handle.cublasHandle,
		C.CUBLAS_OP_N, C.CUBLAS_OP_T,
		C.int(m), C.int(n), C.int(k),
		&alpha,
		x, C.int(m),
		y, C.int(n),
		&beta,
		z, C.int(m)))
	return z
}

func (handle *Handle) Add(x, y *C.float, n int) *C.float {
	if handle.cublasHandle == nil {
		handle.cublasHandle = cublaInit()
	}
	var alpha C.float = 1
	cublasCheck(C.cublasSaxpy(handle.cublasHandle,
		C.int(n),
		&alpha,
		x, 1,
		y, 1))
	return y
}

func (handle *Handle) SumRow(x *C.float, shape []int) *C.float {
	//sum | direction [a,b]
	//    ^           [a,b]
	m := shape[0]
	n := shape[1]
	z := handle.Malloc(m*n)
	if handle.cublasHandle == nil {
		handle.cublasHandle = cublaInit()
	}
	hostSum := make([]float32, m)
	var tmp C.float
	var offset C.size_t = 0
	for i := 0; i < n; i++ {
		xoffset := (*C.float)(unsafe.Pointer((uintptr(offset) + uintptr(unsafe.Pointer(x)))))
		zoffset := unsafe.Pointer((uintptr(offset) + uintptr(unsafe.Pointer(z))))
		cublasCheck(C.cublasSasum(handle.cublasHandle,
			C.int(m),
			xoffset , C.int(1), &tmp))
		for i := 0; i < m; i++ {
			hostSum[i] = float32(tmp)
		}
		C.cudaMemcpy(zoffset, unsafe.Pointer(&hostSum[0]),
			(C.size_t)(unsafe.Sizeof(float32(0))* uintptr(m)), C.cudaMemcpyHostToDevice)
		offset += (C.size_t)(uintptr((C.size_t)(m)) * unsafe.Sizeof(float32(0)))
	}
	return z
}

func (handle *Handle) SumCol(x *C.float, shape []int) *C.float {
	//sum | direction [a,b]
	//    ^           [a,b]
	m := shape[0]
	n := shape[1]
	z := handle.Malloc(m*n)
	if handle.cublasHandle == nil {
		handle.cublasHandle = cublaInit()
	}
	hostSum := make([]float32, m)
	var tmp C.float
	var offset C.size_t = 0
	for i := 0; i < m; i++ {
		xoffset := (*C.float)(unsafe.Pointer((uintptr(offset) + uintptr(unsafe.Pointer(x)))))
		cublasCheck(C.cublasSasum(handle.cublasHandle,
			C.int(n),
			xoffset , C.int(m), &tmp))
		hostSum[i] = float32(tmp)
		offset += (C.size_t)(unsafe.Sizeof(float32(0)))
	}
	for i := 0; i < n; i++ {
		hostSum = append(hostSum, hostSum...)
	}
	C.cudaMemcpy(unsafe.Pointer(z), unsafe.Pointer(&hostSum[0]),
		(C.size_t)(unsafe.Sizeof(float32(0))* uintptr(n*m)), C.cudaMemcpyHostToDevice)
	return z
}

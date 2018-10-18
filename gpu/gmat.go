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
// #cgo LDFLAGS: -L/usr/local/cuda/lib64/ -L/usr/lib/x86_64-linux-gnu -L/root/go/src/github.com/kuroko1t/gmat/gpu/ -lcudart -lcuda -lcudnn -lcublas -lgmat -lcurand
// #include </usr/local/cuda/include/cuda_runtime.h>
// #include </usr/local/cuda/include/cuda.h>
// #include "/root/go/src/github.com/kuroko1t/gmat/gpu/gmat.h"
// #include "cublas_v2.h"
// #include <curand.h>
// #include </usr/include/cudnn.h>
import "C"
import "unsafe"
//import "fmt"

var threadsPerBlock C.int = 256

type Handle struct {
	cublasHandle C.cublasHandle_t
	curandgen C.curandGenerator_t
	stream C.cudaStream_t
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

func (handle *Handle) MakeInit(m, n int, value float32) *C.float {
	z := handle.Malloc(m * n)
	N := C.int(m*n)
	var threadsPerBlock C.int = 256
	var blocksPerGrid C.int =
		(N + threadsPerBlock - 1) / threadsPerBlock
	C.gfill(blocksPerGrid, threadsPerBlock, z, C.float(value))
	return z
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
	defer cudaCheck(C.cudaFree(unsafe.Pointer(gpuptr)))
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
	z := handle.Malloc(1*n)
	if handle.cublasHandle == nil {
		handle.cublasHandle = cublaInit()
	}
	var hostSum C.float
	var offset C.size_t = 0
	var offsetZ C.size_t = 0
	for i := 0; i < n; i++ {
		xoffset := (*C.float)(unsafe.Pointer((uintptr(offset) + uintptr(unsafe.Pointer(x)))))
		zoffset := unsafe.Pointer((uintptr(offsetZ) + uintptr(unsafe.Pointer(z))))
		cublasCheck(C.cublasSasum(handle.cublasHandle,
			C.int(m),
			xoffset , C.int(1), &hostSum))
		C.cudaMemcpy(zoffset, unsafe.Pointer(&hostSum),
			(C.size_t)(unsafe.Sizeof(float32(0))), C.cudaMemcpyHostToDevice)
		offset += (C.size_t)(unsafe.Sizeof(float32(0))* uintptr(m))
		offsetZ += (C.size_t)(unsafe.Sizeof(float32(0)))
	}
	return z
}

func (handle *Handle) SumCol(x *C.float, shape []int) *C.float {
	//sum | direction [a,b]
	//    ^           [a,b]
	m := shape[0]
	n := shape[1]
	z := handle.Malloc(m*1)
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
	C.cudaMemcpy(unsafe.Pointer(z), unsafe.Pointer(&hostSum[0]),
		(C.size_t)(unsafe.Sizeof(float32(0))* uintptr(m)), C.cudaMemcpyHostToDevice)
	return z
}

func (handle *Handle) Mul(x *C.float, y *C.float, shape []int) *C.float {
	m := shape[0]
	n := shape[1]
	z := handle.Malloc(m*n)
	N := C.int(m*n)
	var threadsPerBlock C.int = 256
	var blocksPerGrid C.int =
		(N + threadsPerBlock - 1) / threadsPerBlock
	//fmt.Println(blocksPerGrid)
	C.gmul(blocksPerGrid, threadsPerBlock, x, y, z)
	return z
}

func (handle *Handle) MulE(x *C.float, y float64, shape []int) *C.float {
	m := shape[0]
	n := shape[1]
	z := handle.Malloc(m*n)
	N := C.int(m*n)
	var threadsPerBlock C.int = 256
	var blocksPerGrid C.int =
		(N + threadsPerBlock - 1) / threadsPerBlock
	C.gmule(blocksPerGrid, threadsPerBlock, x, C.float(y), z)
	return z
}

func (handle *Handle) Div(x *C.float, y *C.float, shape []int) *C.float {
	m := shape[0]
	n := shape[1]
	z := handle.Malloc(m*n)
	N := C.int(m*n)
	var blocksPerGrid C.int =
		(N + threadsPerBlock - 1) / threadsPerBlock
	C.gdiv(blocksPerGrid, threadsPerBlock, x, y, z)
	return z
}

func (handle *Handle) Cast(x *C.float, shape []int, castSize int) *C.float {
	m := shape[0]
	n := shape[1]
	stream := handle.StreamInit()
	var offset C.size_t = 0
	if m == 1 {
		var offsetSrc C.size_t = 0
		z := handle.Malloc(castSize * n)
		for j := 0; j < n; j++ {
			xoffset := goffset(x, offsetSrc)
			zoffset := goffset(z, offset)
			var blocksPerGrid C.int =
				(C.int(castSize) + threadsPerBlock - 1) / threadsPerBlock
			C.gdeviceMemset(blocksPerGrid, threadsPerBlock, xoffset, zoffset)
			offset += (C.size_t)(unsafe.Sizeof(float32(0)) * uintptr(castSize))
			offsetSrc += (C.size_t)(unsafe.Sizeof(float32(0)))
		}
		return z
	}
	if n == 1 {
		z := handle.Malloc(m * castSize)
		for i := 0; i < castSize; i++ {
			zoffset := goffset(z, offset)
			C.cudaMemcpyAsync(unsafe.Pointer(zoffset), unsafe.Pointer(x),
				(C.size_t)(unsafe.Sizeof(float32(0)) * uintptr(m)), C.cudaMemcpyDeviceToDevice, stream)
			offset += (C.size_t)(unsafe.Sizeof(float32(0)) * uintptr(castSize))
		}
		return z
	} else {
		z := handle.Malloc(m * n)
		return z
	}
}

func (handle *Handle) Mask(x *C.float, shape []int) *C.float {
	z := handle.Malloc(sizeTensor(shape))
	N := C.int(shape[0] * shape[1])
	var blocksPerGrid C.int =
		(N + threadsPerBlock - 1) / threadsPerBlock
	C.gmask(blocksPerGrid, threadsPerBlock, x, z)
	return z
}

func (handle *Handle) AxpyE(x *C.float, shape []int, b, c float32) *C.float {
	size := sizeTensor(shape)
	z := handle.Malloc(size)
	N := C.int(size)
	var blocksPerGrid C.int =
		(N + threadsPerBlock - 1) / threadsPerBlock
	C.gaxpye(blocksPerGrid, threadsPerBlock, x, C.float(b), C.float(c), z)
	return z
}

func (handle *Handle) Exp(x *C.float, shape []int, b, c float32) *C.float {
	size := sizeTensor(shape)
	z := handle.Malloc(size)
	N := C.int(size)
	var blocksPerGrid C.int =
		(N + threadsPerBlock - 1) / threadsPerBlock
	C.gexp(blocksPerGrid, threadsPerBlock, x, C.float(b), C.float(c), z)
	return z
}

func (handle *Handle) ExpT(x *C.float, shape []int, b, c float32) *C.float {
	size := sizeTensor(shape)
	z := handle.Malloc(size)
	N := C.int(size)
	var blocksPerGrid C.int =
		(N + threadsPerBlock - 1) / threadsPerBlock
	C.gexpT(blocksPerGrid, threadsPerBlock, x, C.float(b), C.float(c), z)
	return z
}

func (handle *Handle) Log(x *C.float, shape []int, b float32) *C.float {
	size := sizeTensor(shape)
	z := handle.Malloc(size)
	N := C.int(size)
	var blocksPerGrid C.int =
		(N + threadsPerBlock - 1) / threadsPerBlock
	C.glog(blocksPerGrid, threadsPerBlock, x, C.float(b), z)
	return z
}

func (handle *Handle) RandomNorm(shape []int) *C.float {
	if handle.curandgen == nil {
		handle.curandgen = curandInit()
	}
	size := sizeTensor(shape)
	z := handle.Malloc(size)
	curandCheck(C.curandGenerateUniform(handle.curandgen, z, C.size_t(size)))
	//curandCheck(C.curandGenerateNormal(handle.curandgen, z, C.size_t(size), 0.0, 0.5))
	return z
}

func (handle *Handle) T(x *C.float, shape []int) *C.float {
	size := sizeTensor(shape)
	m := C.int(shape[1])
	n := C.int(shape[0])
	z := handle.Malloc(size)
	if handle.cublasHandle == nil {
		handle.cublasHandle = cublaInit()
	}
	var alpha C.float = 0
	var beta C.float = 1
	cublasCheck(C.cublasSgeam(handle.cublasHandle,
		C.CUBLAS_OP_N, C.CUBLAS_OP_T,
		m, n,
		&alpha,
		z, m,
		&beta,
		x, n,
		z, m))
	return z
}

func (handle *Handle) Sub(x , y *C.float, shape []int) *C.float {
	size := sizeTensor(shape)
	z := handle.Malloc(size)
	N := C.int(size)
	var blocksPerGrid C.int =
		(N + threadsPerBlock - 1) / threadsPerBlock
	C.gsub(blocksPerGrid, threadsPerBlock, x, y, z)
	return z
}

func (handle *Handle) SqrtT(x *C.float, shape []int, b ,c float32) *C.float {
	size := sizeTensor(shape)
	z := handle.Malloc(size)
	N := C.int(size)
	var blocksPerGrid C.int =
		(N + threadsPerBlock - 1) / threadsPerBlock
	C.gsqrtT(blocksPerGrid, threadsPerBlock, x, C.float(b), C.float(c), z)
	return z
}

func (handle *Handle) ArgMaxCol(x *C.float, shape []int) *C.float {
	// max col
	size := sizeTensor(shape)
	z := handle.Malloc(size)
	if handle.cublasHandle == nil {
		handle.cublasHandle = cublaInit()
	}
	m := shape[0]
	n := shape[1]
	tmpSum := handle.Malloc(m)
	var tmp C.int
	var offset C.size_t = 0
	for i := 0; i < m; i++ {
		xoffset := goffset(x, offset)
		sumoffset := goffset(tmpSum, offset)
		cublasCheck(C.cublasIsamax(handle.cublasHandle,
			C.int(n),
			xoffset , C.int(m), &tmp))
		tmp = tmp -1
		xoffsetMax := goffset(xoffset, C.size_t(unsafe.Sizeof(float32(0)) * uintptr(tmp* C.int(m))))
		offset += (C.size_t)(unsafe.Sizeof(float32(0)))
		C.cudaMemcpy(unsafe.Pointer(sumoffset), unsafe.Pointer(xoffsetMax),
			(C.size_t)(unsafe.Sizeof(float32(0))), C.cudaMemcpyDeviceToDevice)
	}
	offset = 0
	for i := 0; i < n; i++ {
		zoffset := goffset(z, offset)
		C.cudaMemcpy(unsafe.Pointer(zoffset), unsafe.Pointer(tmpSum),
			(C.size_t)(unsafe.Sizeof(float32(0)) * uintptr(m)), C.cudaMemcpyDeviceToDevice)
		offset += (C.size_t)(unsafe.Sizeof(float32(0)) * uintptr(m))
	}
	return z
}

func goffset(x *C.float, offset C.size_t) *C.float {
	xoffset := (*C.float)(unsafe.Pointer((uintptr(offset) + uintptr(unsafe.Pointer(x)))))
	return xoffset
}

func (handle *Handle) Sum(x *C.float, shape []int) float64 {
	size := sizeTensor(shape)
	if handle.cublasHandle == nil {
		handle.cublasHandle = cublaInit()
	}
	var result C.float = 0
	cublasCheck(C.cublasSasum(
		handle.cublasHandle,
		C.int(size),
		x, 1, &result ))
	return float64(result)
}

func (handle *Handle) Max(x *C.float, shape []int) float64 {
	size := sizeTensor(shape)
	if handle.cublasHandle == nil {
		handle.cublasHandle = cublaInit()
	}
	var index C.int = 0
	cublasCheck(C.cublasIsamax(
		handle.cublasHandle,
		C.int(size),
		x, 1, &index))
	xoffset := goffset(x, C.size_t(uintptr(index - 1) * unsafe.Sizeof(float32(0))))
	var result C.float
	C.cudaMemcpy(unsafe.Pointer(&result), unsafe.Pointer(xoffset),
			(C.size_t)(unsafe.Sizeof(float32(0))), C.cudaMemcpyDeviceToHost)
	return float64(result)
}

// +build gpu

package gpu

// #cgo CFLAGS: -I/usr/local/cuda-9.1/targets/x86_64-linux/include/
// #cgo LDFLAGS: -L/usr/local/cuda/lib64/ -L/usr/lib/x86_64-linux-gnu -lcudart -lcuda -lcudnn -lcublas
// #include </usr/local/cuda/include/cuda_runtime.h>
// #include "cublas_v2.h"
// #include </usr/include/cudnn.h>
import "C"
import "unsafe"

type Handle struct {
	cublasHandle C.cublasHandle_t
}

type Tensor struct {
	CPU   [][]float64
	CPU4D [][][][]float64
	GPU   *C.float
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

func (handle *Handle) SumRow(x, y *C.float, n int) *C.float {
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

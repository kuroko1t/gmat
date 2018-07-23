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

func (handle *Handle) Make2D(m, n int) *C.float {
	var z *C.float
	gpuptr := unsafe.Pointer(z)
	typeSize := int(unsafe.Sizeof(float32(0)))
	C.cudaMalloc(&gpuptr, C.size_t(m*n*typeSize))
	return (*C.float)(gpuptr)
}

func (handle *Handle) Dot(x, y *C.float, m, n, k int) *C.float {
	z := handle.Make2D(m, n)
	if handle.cublasHandle == nil {
		handle.cublasHandle = cublaInit()
	}
	cuDot(handle.cublasHandle, C.int(m), C.int(n), C.int(k), x, y, z)
	return z
}

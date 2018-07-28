// +build gpu

package gpu

// #cgo CFLAGS: -I/usr/local/cuda-9.1/targets/x86_64-linux/include/
// #cgo LDFLAGS: -L/usr/local/cuda/lib64/ -L/usr/lib/x86_64-linux-gnu -lcudart -lcuda -lcudnn -lcublas
// #include </usr/local/cuda/include/cuda_runtime.h>
// #include "cublas_v2.h"
// #include </usr/include/cudnn.h>
import "C"
import "unsafe"
import "fmt"

type Handle struct {
	cublasHandle C.cublasHandle_t
}

type Tensor struct {
	CPU   [][]float64
	CPU4D [][][][]float64
	GPU   *C.float
	Shape []int
}

func (handle *Handle) Malloc2D(n, m int) *C.float {
	var z *C.float
	gpuptr := unsafe.Pointer(z)
	typeSize := int(unsafe.Sizeof(float32(0)))
	gpuSize := C.size_t(m * n * typeSize)
	fmt.Println("malloc gpuSize", gpuSize)
	C.cudaMalloc(&gpuptr, gpuSize)
	return (*C.float)(gpuptr)
}

func (handle *Handle) CopyH2D(x [][]float64) (int, int, *C.float) {
	n := len(x)
	m := len(x[0])
	var z *C.float
	gpuptr := unsafe.Pointer(z)
	typeSize := int(unsafe.Sizeof(float32(0)))
	//fmt.Println("typeSize", typeSize)
	gpuSize := C.size_t(m * n * typeSize)
	fmt.Println("n,m:", n, ",", m, "h2d gpuSize:", gpuSize, "typeSize:", typeSize)
	C.cudaMalloc(&gpuptr, gpuSize)
	x32 := d2f(x)
	C.cudaMemcpy(gpuptr, unsafe.Pointer(&x32[0]), gpuSize, C.cudaMemcpyHostToDevice)
	return n, m, (*C.float)(gpuptr)
}

func (handle *Handle) CopyD2H(shape []int, gpuptr *C.float) [][]float64 {
	fmt.Println("cpy d2h shape", shape)
	n := shape[0]
	m := shape[1]
	z := make([][]float32, n*m)
	//for i := 0; i < n; i++ {
	// 	z[i] = make([]float32, m)
	//}
	typeSize := int(unsafe.Sizeof(float32(0)))
	fmt.Println("n,m:", n, ",", m, "typeSize:", typeSize)
	cudaCheck(C.cudaMemcpy(unsafe.Pointer(&z[0]),
		unsafe.Pointer(gpuptr),
		C.size_t(n*m*typeSize), C.cudaMemcpyDeviceToHost))
	z64 := f2d(z)
	fmt.Println("cpy d2h", z)
	return z64
}

func (handle *Handle) Dot(x, y *C.float, m, n, k int) *C.float {
	z := handle.Malloc2D(m, n)
	if handle.cublasHandle == nil {
		handle.cublasHandle = cublaInit()
	}
	cuDot(handle.cublasHandle, C.int(m), C.int(n), C.int(k), x, y, z)
	return z
}

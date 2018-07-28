// +build gpu

package gpu

// #cgo CFLAGS: -I/usr/local/cuda-9.1/targets/x86_64-linux/include/
// #cgo LDFLAGS: -L/usr/local/cuda/lib64/ -L/usr/lib/x86_64-linux-gnu -lcudart -lcuda -lcudnn -lcublas
// #include </usr/local/cuda/include/cuda_runtime.h>
// #include "cublas_v2.h"
// #include </usr/include/cudnn.h>
import "C"

//import "unsafe"
import "fmt"
import "runtime"

func d2f(x [][]float64) []float32 {
	n := len(x)
	m := len(x[0])
	z := make([]float32, n)
	//for i := 0; i < n; i++ {
	// 	z[i] = make([]float32, m)
	//}
	for i := range x {
		for j := range x[i] {
			z[(i+1)*j] = float32(x[i][j])
		}
	}
	return z
}

func f2d(x []float32) [][]float64 {
	n := len(x)
	m := len(x[0])
	z := make([][]float64, n)
	for i := 0; i < n; i++ {
		z[i] = make([]float64, m)
	}
	for i := range z {
		for j := range z[i] {
			z[i][j] = float64(x[i][j])
		}
	}
	return z
}

func cublaInit() C.cublasHandle_t {
	var cublasHandle C.cublasHandle_t
	cublasCheck(C.cublasCreate(&cublasHandle))
	return cublasHandle
}

func cudaCheck(error C.cudaError_t) {
	if error != 0 {
		er_message := C.GoString(C.cudaGetErrorString(error))
		fmt.Print("ERR:", er_message, "  ")
		_, file, line, _ := runtime.Caller(1)
		fmt.Println(file, line)
	}
}

func cublasCheck(error C.cublasStatus_t) {
	if error != C.CUBLAS_STATUS_SUCCESS {
		er_message := "cublas error"
		fmt.Print("ERR:", er_message, "  ")
		_, file, line, _ := runtime.Caller(1)
		fmt.Println(file, line)
	}
}

func cuDot(handle C.cublasHandle_t, m, n, k C.int, A_dev, B_dev, C_dev *C.float) {
	var alpha C.float = 1
	var beta C.float = 1
	cublasCheck(C.cublasSgemm(handle,
		C.CUBLAS_OP_N, C.CUBLAS_OP_N,
		m, n, k,
		&alpha,
		A_dev, m,
		B_dev, k,
		&beta,
		C_dev, m))
}

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
import "log"

func d2f(x [][]float64) []float32 {
	n := len(x)
	m := len(x[0])
	z := make([]float32, n*m)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			z[j+i*n] = float32(x[j][i])
		}
	}
	return z
}

func f2d(x []float32, n, m int) [][]float64 {
	if len(x) != n*m {
		log.Fatal("not match shape float ,double slice")
	}
	z := make([][]float64, n)
	for i := 0; i < n; i++ {
		z[i] = make([]float64, m)
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			z[j][i] = float64(x[j+i*n])
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

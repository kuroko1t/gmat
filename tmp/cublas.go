package main

// #cgo CFLAGS: -I/usr/local/cuda-9.1/targets/x86_64-linux/include/
// #cgo LDFLAGS: -L/usr/local/cuda/lib64/ -L/usr/lib/x86_64-linux-gnu -lcudart -lcuda -lcudnn -lcublas
// #include </usr/local/cuda/include/cuda_runtime.h>
// #include "cublas_v2.h"
// #include </usr/include/cudnn.h>
import "C"
import "unsafe"
import "fmt"
import "runtime"

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

func Host2GPU(input []float32) *C.float {
	var in_data_dev *C.float
	dataSize := len(input)
	typeSize := int(unsafe.Sizeof(float32(0)))
	size := C.size_t(dataSize * typeSize)
	gpuptr := unsafe.Pointer(in_data_dev)
	C.cudaMalloc(&gpuptr, size)
	C.cudaMemcpy(gpuptr, unsafe.Pointer(&input[0]), size, C.cudaMemcpyHostToDevice)
	return (*C.float)(gpuptr)
}

func GPU2Host(dev_data unsafe.Pointer, elemNum int) []float32 {
	output := make([]float32, elemNum)
	typeSize := int(unsafe.Sizeof(float32(0)))
	cudaCheck(C.cudaMemcpy(unsafe.Pointer(&output[0]),
		dev_data,
		C.size_t(elemNum*typeSize), C.cudaMemcpyDeviceToHost))
	//cudaCheck(C.cudaFree(dev_data))
	return output
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

func cuAdd(handle C.cublasHandle_t, n C.int, A_dev, B_dev *C.float) {
	var alpha C.float = 1
	cublasCheck(C.cublasSaxpy(handle,
		n,
		&alpha,
		A_dev, 1,
		B_dev, 1))
}

func cublaInit() C.cublasHandle_t {
	var cublasHandle C.cublasHandle_t
	cublasCheck(C.cublasCreate(&cublasHandle))
	return cublasHandle
}

func cublasDestroy(cublasHandle *C.cublasHandle_t) {
	cublasCheck(C.cublasDestroy(*cublasHandle))
}

func cuFree(A_dev *C.float) {
	C.cudaFree(unsafe.Pointer(A_dev))
}

func main() {
	//cudaStatus = cudaSetDevices(0)
	var m C.int = 2
	var k C.int = 2
	var n C.int = 2
	cudaCheck(C.cudaSetDevice(0))
	A := []float32{
		1.0, 1.0,
		3, 2,
	}
	B := []float32{
		1.0, 2.0,
		3.0, 4.0,
	}
	C := make([]float32, int(m*n))
	CAdd := []float32{
		2, 3,
		4, 5,
	}
	A_dev := Host2GPU(A)
	B_dev := Host2GPU(B)
	C_dev := Host2GPU(C)
	CAdd_dev := Host2GPU(CAdd)
	cublasHandle := cublaInit()
	cuDot(cublasHandle, m, n, k, A_dev, B_dev, C_dev)
	cuAdd(cublasHandle, n*m, A_dev, CAdd_dev)
	outData := GPU2Host(unsafe.Pointer(C_dev), int(m*n))
	outDataAdd := GPU2Host(unsafe.Pointer(CAdd_dev), int(m*n))
	fmt.Println("outData", outData)
	fmt.Println("outAdd", outDataAdd)
	cuFree(A_dev)
	cuFree(B_dev)
	cuFree(C_dev)
	cublasDestroy(&cublasHandle)
	//cudaCheck(C.cudaFree(unsafe.Pointer(in_unsafe)))
	//cudaCheck(C.cudaFree(unsafe.Pointer(w_unsafe)))
}

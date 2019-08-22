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
// #cgo LDFLAGS: -L/usr/local/cuda/lib64/ -L/usr/lib/x86_64-linux-gnu -lcudart -lcuda -lcudnn -lcublas -lcurand
// #include </usr/local/cuda/include/cuda_runtime.h>
// #include "cublas_v2.h"
// #include <curand.h>
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

func curandInit() C.curandGenerator_t {
	var curandgen C.curandGenerator_t
	curandCheck(C.curandCreateGenerator(&curandgen, C.CURAND_RNG_PSEUDO_DEFAULT))
	curandCheck(C.curandSetPseudoRandomGeneratorSeed(curandgen, 0))
	return curandgen
}

func (handle *Handle) StreamInit() C.cudaStream_t {
	var stream C.cudaStream_t
	C.cudaStreamCreate(&stream)
	return stream
}

func cudaCheck(error C.cudaError_t) {
	if error != 0 {
		er_message := C.GoString(C.cudaGetErrorString(error))
		fmt.Print("ERR:", er_message, "  ")
		_, file, line, _ := runtime.Caller(1)
		fmt.Println(file, line)
	}
}

func curandCheck(error C.curandStatus_t) {
	if error != C.CURAND_STATUS_SUCCESS {
		fmt.Print("ERR:", error, "  ")
		_, file, line, _ := runtime.Caller(1)
		fmt.Println(file, line)
	}
}

func cublasCheck(error C.cublasStatus_t) {
	if error != C.CUBLAS_STATUS_SUCCESS {
		fmt.Print("ERR:", error, "  ")
		_, file, line, _ := runtime.Caller(1)
		fmt.Println(file, line)
	}
}

func sizeTensor(x []int) int {
	sum := 1
	for i := range x {
		sum *= x[i]
	}
	return sum
}

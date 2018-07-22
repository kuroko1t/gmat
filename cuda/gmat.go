package gpu

// #cgo CFLAGS: -I/usr/local/cuda-9.0/targets/x86_64-linux/include/
// #cgo LDFLAGS: -L/usr/local/cuda-8.0/targets/x86_64-linux/lib/ -L/usr/lib/x86_64-linux-gnu -lcudart -lcuda -lcudnn -lcublas
// #include </usr/include/cuda_runtime.h>
// #include "cublas_v2.h"
// #include </usr/local/cuda/targets/x86_64-linux/include/cudnn.h>
import (
	"C"
	import "unsafe"
)


type Handle struct {
	cublasHandle C.cublasHandle_t
}

func (handle * Handle) Dot(x , y *C.float) *C.float {
	cuDot(handle.cublasHandle, )
}

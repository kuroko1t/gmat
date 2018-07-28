// +build gpu

package gmat

import (
	"C"
	"fmt"
	"github.com/kuroko1t/gmat/gpu"
	"log"
)

type Tensor gpu.Tensor

var handle = &gpu.Handle{}

func Make2D(n, m int) (z Tensor) {
	z.GPU = handle.Malloc2D(n, m)
	z.Shape = []int{n, m}
	return z
}

func CopyH2D(x [][]float64) (z Tensor) {
	n, m, gpuptr := handle.CopyH2D(x)
	z.GPU = gpuptr
	z.Shape = []int{n, m}
	return z
}

func CopyD2H(z *Tensor) {
	(*z).CPU = handle.CopyD2H((*z).Shape, (*z).GPU)
	fmt.Println((*z).CPU)
}

func Dot(x, y Tensor) (z Tensor) {
	m, kx := x.Shape[0], x.Shape[1]
	ky, n := y.Shape[0], y.Shape[1]
	if ky != kx {
		log.Fatal("mismatch input shape")
	}
	z.GPU = handle.Dot(x.GPU, y.GPU, m, n, kx)
	z.Shape = []int{m, n}
	return z
}

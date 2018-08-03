// +build gpu

package gmat

import (
	"C"
	"github.com/kuroko1t/gmat/gpu"
	"log"
)

type Tensor gpu.Tensor

var handle = &gpu.Handle{}

func Make2D(n, m int) (z Tensor) {
	z.GPU = handle.Malloc(n * m)
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
}

func Dot(x, y Tensor) (z Tensor) {
	m, kx := x.Shape[0], x.Shape[1]
	ky, n := y.Shape[0], y.Shape[1]
	if ky != kx {
		log.Fatal("Dot mismatch input shape")
	}
	z.GPU = handle.Dot(x.GPU, y.GPU, m, n, kx)
	z.Shape = []int{m, n}
	return z
}

func TDot(x, y Tensor) (z Tensor) {
	kx, m := x.Shape[0], x.Shape[1]
	ky, n := y.Shape[0], y.Shape[1]
	if ky != kx {
		log.Fatal("Dot mismatch input shape")
	}
	z.GPU = handle.TDot(x.GPU, y.GPU, m, n, kx)
	z.Shape = []int{m, n}
	return z
}

func DotT(x, y Tensor) (z Tensor) {
	m, kx := x.Shape[0], x.Shape[1]
	n, ky := y.Shape[0], y.Shape[1]
	if ky != kx {
		log.Fatal("Dot mismatch input shape")
	}
	z.GPU = handle.DotT(x.GPU, y.GPU, m, n, kx)
	z.Shape = []int{m, n}
	return z
}

func Add(x, y Tensor) (z Tensor) {
	mx, nx := x.Shape[0], x.Shape[1]
	my, ny := y.Shape[0], y.Shape[1]
	if (mx != my) || (nx != ny) {
		log.Fatal("Add mismatch input shape")
	}
	z.GPU = handle.Add(x.GPU, y.GPU, mx*nx)
	z.Shape = []int{mx, nx}
	return z
}

func SumRow(x Tensor) (z Tensor) {
	z.GPU = handle.SumRow(x.GPU, x.Shape)
	z.Shape = x.Shape
	return z
}

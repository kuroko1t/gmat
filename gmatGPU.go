// +build gpu

package gmat

import (
	"C"
	"github.com/kuroko1t/gmat/gpu"
	"log"
)

type Tensor gpu.Tensor

func Dot(x, y Tensor) (z Tensor) {
	m, kx := x.Shape[0], x.Shape[1]
	ky, n := y.Shape[0], y.Shape[1]
	if ky != kx {
		log.Fatal("mismatch input shape")
	}
	handle := &gpu.Handle{}
	z.GPU = handle.Dot(x.GPU, y.GPU, m, n, kx)
	return z
}

// +build gpu

package gmat

import (
	//"github.com/kuroko1t/gmat/data"
	"github.com/kuroko1t/gmat/gpu"
	"log"
)

//type Data data.Data

func Dot(x, y Data) Data {
	m, kx := x.Shape[0], x.Shape[1]
	ky, n := y.Shape[0], y.Shape[1]
	if ky != kx {
		log.Fatal("mismatch input shape")
	}
	handle := &gpu.Handle{}
	z := handle.Dot(x.GPU, y.GPU, m, n, kx)
	return z
}

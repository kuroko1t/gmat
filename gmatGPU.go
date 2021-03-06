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

func Make2DInitArray(x [][]float64) (z Tensor) {
	n := 0
	m := 0
	n, m, z.GPU = handle.CopyH2D(x)
	z.Shape = []int{n, m}
	return z
}

func MakeInit(n, m int, value float64) (z Tensor) {
	z.GPU = handle.MakeInit(n, m, float32(value))
	z.Shape = []int{n, m}
	return z
}

func Shape2D(x Tensor) (int, int) {
	return x.Shape[0], x.Shape[1]
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
	z.Shape = []int{1, x.Shape[1]}
	return z
}

func SumCol(x Tensor) (z Tensor) {
	z.GPU = handle.SumCol(x.GPU, x.Shape)
	z.Shape = []int{x.Shape[0], 1}
	return z
}

func MulE(x Tensor, y float64) (z Tensor) {
	z.GPU = handle.MulE(x.GPU, y, x.Shape)
	z.Shape = x.Shape
	return z
}

func Mul(x, y Tensor) (z Tensor) {
	z.GPU = handle.Mul(x.GPU, y.GPU, x.Shape)
	z.Shape = x.Shape
	return z
}

func Div(x, y Tensor) (z Tensor) {
	z.GPU = handle.Div(x.GPU, y.GPU, x.Shape)
	z.Shape = x.Shape
	return z
}

func Cast(x Tensor, castSize int) (z Tensor) {
	if (x.Shape[0] != 1) && (x.Shape[1] != 1) {
		log.Fatal("Cast.not support format")
	}
	if x.Shape[0] == 1 {
		z.Shape = []int{castSize, x.Shape[1]}
	}
	if x.Shape[1] == 1 {
		z.Shape = []int{x.Shape[0], castSize}
	}
	z.GPU = handle.Cast(x.GPU, x.Shape, castSize)
	return z
}

func Mask(x Tensor) (z Tensor) {
	z.GPU = handle.Mask(x.GPU, x.Shape)
	z.Shape = x.Shape
	return z
}

func AxpyE(x Tensor, b, c float64) (z Tensor) {
	// d[i] = a[i] *b + c
	z.GPU = handle.AxpyE(x.GPU, x.Shape, float32(b), float32(c))
	z.Shape = x.Shape
	return z
}

func Exp(x Tensor, b, c float64) (z Tensor) {
	// d[i] = expf(a[i] * b) + c;
	z.GPU = handle.Exp(x.GPU, x.Shape, float32(b), float32(c))
	z.Shape = x.Shape
	return z
}

func ExpT(x Tensor, b, c float64) (z Tensor) {
	// d[i] = 1/ (expf(a[i] * b) + c);
	z.GPU = handle.ExpT(x.GPU, x.Shape, float32(b), float32(c))
	z.Shape = x.Shape
	return z
}

func Log(x Tensor, b float64) (z Tensor) {
	// c[i] = logf(a[i] + b);
	z.GPU = handle.Log(x.GPU, x.Shape, float32(b))
	z.Shape = x.Shape
	return z
}

func RandomNorm(size []int) (z Tensor) {
	z.GPU = handle.RandomNorm(size)
	z.Shape = size
	return z
}

func T(x Tensor) (z Tensor) {
	// Transpose Tensor
	z.GPU = handle.T(x.GPU, x.Shape)
	z.Shape = []int{x.Shape[1], x.Shape[0]}
	return z
}

func Sub(x, y Tensor) (z Tensor) {
	// c[i] = a[i] - b[i];
	z.GPU = handle.Sub(x.GPU, y.GPU, x.Shape)
	z.Shape = x.Shape
	return z
}

func SqrtT(x Tensor, b, c float64) (z Tensor) {
	// c[i] = 1 / (sqrtf(a[i] + b) + d);
	z.GPU = handle.SqrtT(x.GPU, x.Shape, float32(b), float32(c))
	z.Shape = x.Shape
	return z
}

func ArgMaxCol(x Tensor) (z Tensor) {
	z.GPU = handle.ArgMaxCol(x.GPU, x.Shape)
	z.Shape = x.Shape
	return z
}

func Sum(x Tensor) float64 {
	sum := handle.Sum(x.GPU, x.Shape)
	return sum
}

func Max(x Tensor) float64 {
	maxvalue := handle.Max(x.GPU, x.Shape)
	return maxvalue
}

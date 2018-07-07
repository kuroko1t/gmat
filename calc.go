package gmat

import (
	"log"
	"math/rand"
)

func Make(n int, m int) ([][]float64) {
	z := make([][]float64, n)
	for i:=0; i<n; i++{
		z[i] = make([]float64, m)
	}
	return z
}

func MakeInit(n int, m int, value float64) ([][]float64) {
	z := make([][]float64, n)
	for i:=0; i<n; i++{
		z[i] = make([]float64, m)
	}
	for i, zArray := range z {
		for j, _ := range zArray {
			z[i][j] = value
		}
	}
	return z
}

func Add(x [][]float64, y[][]float64) ([][]float64) {
	n := len(x)
	m := len(x[0])
	z := make([][]float64, n)
	for i:=0; i<n; i++{
		z[i] = make([]float64, m)

	}
	for i, xArray := range x {
		for j, _ := range xArray {
			z[i][j] = x[i][j] + y[i][j]
		}
	}
	return z
}

func Sub(x [][]float64, y[][]float64) ([][]float64) {
	n := len(x)
	m := len(x[0])
	z := make([][]float64, n)
	for i:=0; i<n; i++{
		z[i] = make([]float64, m)

	}
	for i, xArray := range x {
		for j, _ := range xArray {
			z[i][j] = x[i][j] - y[i][j]
		}
	}
	return z
}

func MulE(x [][]float64, y float64) ([][]float64) {
	n := len(x)
	m := len(x[0])
	z := make([][]float64, n)
	for i:=0; i<n; i++{
		z[i] = make([]float64, m)
	}
	for i, xArray := range x {
		for j, _ := range xArray {
			z[i][j] = x[i][j] * y
		}
	}
	return z
}

func Mul(x [][]float64, y [][]float64) ([][]float64) {
	n := len(x)
	m := len(x[0])
	z := make([][]float64, n)
	for i:=0; i<n; i++{
		z[i] = make([]float64, m)
	}
	for i, xArray := range x {
		for j, _ := range xArray {
			z[i][j] = x[i][j] * y[i][j]
		}
	}
	return z
}

func Div(x [][]float64, y [][]float64) ([][]float64) {
	n := len(x)
	m := len(x[0])
	z := make([][]float64, n)
	for i:=0; i<n; i++{
		z[i] = make([]float64, m)
	}
	for i, xArray := range x {
		for j, _ := range xArray {
			z[i][j] = x[i][j] / y[i][j]
		}
	}
	return z
}

func T(x [][]float64) ([][]float64) {
	n := len(x)
	m := len(x[0])
	z := make([][]float64, m)
	for i:=0; i<m; i++{
		z[i] = make([]float64, n)
	}
	for i, zArray := range z {
		for j, _ := range zArray {
			z[i][j] = x[j][i]
		}
	}
	return z
}


func Apply(x [][]float64, fn func(float64) (float64)) ([][]float64) {
	n := len(x)
	m := len(x[0])
	z := make([][]float64, n)
	for i:=0; i<n; i++{
		z[i] = make([]float64, m)
	}
	for i, xArray := range x {
		for j, _ := range xArray {
			z[i][j] = fn(x[i][j])
		}
	}
	return z
}

func Dot(x [][]float64, y[][]float64) ([][]float64) {
	nx := len(x)
	mx := len(x[0])
	ny := len(y)
	my := len(y[0])
	if mx != ny {
		log.Fatal("mismatch matrix number")
	}
	z := make([][]float64, nx)
	for i:=0; i<nx; i++{
		z[i] = make([]float64, my)

	}
	zElem := 0.0
	for zcol:= 0; zcol < nx ; zcol++ {
		for zrow:= 0; zrow < my ; zrow++ {
			for i:= 0; i < mx ; i++ {
				zElem += x[zcol][i] * y[i][zrow]
			}
			z[zcol][zrow] = zElem
			zElem = 0
		}
	}
	return z
}

func SumRow(x [][]float64) ([][]float64) {
	n := len(x)
	m := len(x[0])
	sumArray := make([][]float64, n)
	for i:=0; i<n; i++{
		sumArray[i] = make([]float64, m)
	}
	for j := 0; j < m; j++ {
		sumValue := 0.0
		for i := 0; i < n; i++ {
			sumValue += x[i][j]
		}
		for i := 0; i < n; i++ {
			sumArray[i][j] = sumValue
		}
	}
	return sumArray
}

func SumCol(x [][]float64) ([][]float64) {
	n := len(x)
	m := len(x[0])
	sumArray := make([][]float64, n)
	for i:=0; i<n; i++{
		sumArray[i] = make([]float64, m)
	}
	for j := 0; j < n; j++ {
		sumValue := 0.0
		for i := 0; i < m; i++ {
			sumValue += x[j][i]
		}
		for i := 0; i < m; i++ {
			sumArray[j][i] = sumValue
		}
	}
	return sumArray
}

func RandomArray(r int, c int, init float64) [][]float64 {
	z := make([][]float64, r)
	for i:=0; i<r; i++{
		z[i] = make([]float64, c)
	}
	for i, zArray := range z {
		for j, _ := range zArray {
			z[i][j] = rand.NormFloat64() * init
		}
	}
	return z
}

package gmat

import "log"

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
	for zcol:= 0; zcol < ny ; zcol++ {
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

// +build gpu

package gmat

import (
	"fmt"
	"testing"
)

func ExpCheck(zReal [][]float64, zExp [][]float64, t *testing.T) {
	success := true
	for i, zArray := range zExp {
		for j, _ := range zArray {
			if zExp[i][j] != zReal[i][j] {
				success = false
			}
		}
	}
	if !success {
		fmt.Println("Real:", zReal)
		fmt.Println("Exp:", zExp)
		t.Fatal("failed Test!")
	}
}

func TestDotSuccess(t *testing.T) {
	xdot := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}
	ydot := [][]float64{
		{8, 7},
		{5, 4},
		{2, 1},
	}
	zExp := [][]float64{
		{24, 18},
		{69, 54},
		{114, 90},
	}
	xdotGPU := CopyH2D(xdot)
	ydotGPU := CopyH2D(ydot)
	zRealGPU := Dot(xdotGPU, ydotGPU)
	CopyD2H(&zRealGPU)
	zReal := zRealGPU.CPU
	ExpCheck(zReal, zExp, t)
}

func TestTDotSuccess(t *testing.T) {
	xdot := [][]float64{
		{1, 4, 7},
		{2, 5, 8},
		{3, 6, 9},
	}
	ydot := [][]float64{
		{8, 7},
		{5, 4},
		{2, 1},
	}
	zExp := [][]float64{
		{24, 18},
		{69, 54},
		{114, 90},
	}
	xdotGPU := CopyH2D(xdot)
	ydotGPU := CopyH2D(ydot)
	zRealGPU := TDot(xdotGPU, ydotGPU)
	CopyD2H(&zRealGPU)
	zReal := zRealGPU.CPU
	ExpCheck(zReal, zExp, t)
}

func TestDotTSuccess(t *testing.T) {
	xdot := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}
	ydot := [][]float64{
		{8, 5, 2},
		{7, 4, 1},
	}
	zExp := [][]float64{
		{24, 18},
		{69, 54},
		{114, 90},
	}
	xdotGPU := CopyH2D(xdot)
	ydotGPU := CopyH2D(ydot)
	zRealGPU := DotT(xdotGPU, ydotGPU)
	CopyD2H(&zRealGPU)
	zReal := zRealGPU.CPU
	ExpCheck(zReal, zExp, t)
}

func TestAddSuccess(t *testing.T) {
	xdot := [][]float64{
		{1, 2},
		{4, 5},
		{7, 8},
	}
	ydot := [][]float64{
		{8, 7},
		{5, 4},
		{2, 1},
	}
	zExp := [][]float64{
		{9, 9},
		{9, 9},
		{9, 9},
	}
	xdotGPU := CopyH2D(xdot)
	ydotGPU := CopyH2D(ydot)
	zRealGPU := Add(xdotGPU, ydotGPU)
	CopyD2H(&zRealGPU)
	zReal := zRealGPU.CPU
	ExpCheck(zReal, zExp, t)
}

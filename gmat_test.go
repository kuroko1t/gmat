// +build gpu

package gmat

import (
	//"github.com/kuroko1t/gmat/cpu"
	//"github.com/kuroko1t/gmat/gpu"
	//"github.com/kuroko1t/gmat/data"
	//"github.com/kuroko1t/gmat"
	//"log"
	//"math"
	//"math/rand"
	//"sync"
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
	//zReal := cpu.Dot(xdot, ydot)
	// test GPU
	fmt.Println(xdot)
	xdotGPU := CopyH2D(xdot)
	CopyD2H(&xdotGPU)
	fmt.Println("xdot", xdotGPU.CPU)
	ydotGPU := CopyH2D(ydot)
	zRealGPU := Dot(xdotGPU, ydotGPU)
	CopyD2H(&zRealGPU)
	zReal := zRealGPU.CPU
	fmt.Println(zReal)
	ExpCheck(zReal, zExp, t)
	//ExpCheck(zRealGPU, zExp, t)
}

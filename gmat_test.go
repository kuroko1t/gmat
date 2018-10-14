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

func ExpRangeCheck(zReal [][]float64, start, end float64, t *testing.T) {
	success := true
	for i, zArray := range zReal {
		for j, _ := range zArray {
			if start > zReal[i][j] || end < zReal[i][j] {
				success = false
			}
		}
	}
	if !success {
		fmt.Println("Real:", zReal)
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

func TestSumRowSuccess(t *testing.T) {
	zExp := [][]float64{
		{12, 15, 18},
		{12, 15, 18},
		{12, 15, 18},
	}
	x := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
		// 12 ,15, 18
	}
	xGPU := CopyH2D(x)
	zRealGPU := SumRow(xGPU)
	CopyD2H(&zRealGPU)
	zReal := zRealGPU.CPU
	ExpCheck(zReal, zExp, t)
}

func TestSumColSuccess(t *testing.T) {
	zExp := [][]float64{
		{6, 6, 6},
		{15, 15, 15},
		{24, 24, 24},
	}
	x := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
		// 12 ,15, 18
	}
	xGPU := CopyH2D(x)
	zRealGPU := SumCol(xGPU)
	CopyD2H(&zRealGPU)
	zReal := zRealGPU.CPU
	ExpCheck(zReal, zExp, t)
}

func TestMulSuccess(t *testing.T) {
	x := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}
	y := [][]float64{
		{1, 3, 2},
		{1, 3, 2},
		{1, 3, 2},
	}
	zExp := [][]float64{
		{1, 6, 6},
		{4, 15, 12},
		{7, 24, 18},
	}
	xGPU := CopyH2D(x)
	yGPU := CopyH2D(y)
	zRealGPU := Mul(xGPU, yGPU)
	CopyD2H(&zRealGPU)
	zReal := zRealGPU.CPU
	ExpCheck(zReal, zExp, t)
}

func TestMulESuccess(t *testing.T) {
	x := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}
	y := 4.0
	zExp := [][]float64{
		{4, 8, 12},
		{16, 20, 24},
		{28, 32, 36},
	}
	xGPU := CopyH2D(x)
	zRealGPU := MulE(xGPU, y)
	CopyD2H(&zRealGPU)
	zReal := zRealGPU.CPU
	ExpCheck(zReal, zExp, t)
}

func TestDivSuccess(t *testing.T) {
	var x = [][]float64{
		{4, 6, 8},
		{4, 6, 8},
		{4, 6, 8},
	}
	var y = [][]float64{
		{2, 3, 4},
		{2, 3, 4},
		{2, 3, 4},
	}
	zExp := [][]float64{
		{2, 2, 2},
		{2, 2, 2},
		{2, 2, 2},
	}
	xGPU := CopyH2D(x)
	yGPU := CopyH2D(y)
	zRealGPU := Div(xGPU, yGPU)
	CopyD2H(&zRealGPU)
	zReal := zRealGPU.CPU
	ExpCheck(zReal, zExp, t)
}

func TestCastSuccess(t *testing.T) {
	x1 := [][]float64{
		{12, 15, 18},
	}
	z1Exp := [][]float64{
		{12, 15, 18},
		{12, 15, 18},
	}
	x1GPU := CopyH2D(x1)
	z1RealGPU := Cast(x1GPU, 2)
	CopyD2H(&z1RealGPU)
	z1Real := z1RealGPU.CPU
	ExpCheck(z1Real, z1Exp, t)
	x2 := [][]float64{
		{12},
		{15},
		{18},
	}
	z2Exp := [][]float64{
		{12, 12, 12},
		{15, 15, 15},
		{18, 18, 18},
	}
	x2GPU := CopyH2D(x2)
	z2RealGPU := Cast(x2GPU, 3)
	CopyD2H(&z2RealGPU)
	z2Real := z2RealGPU.CPU
	ExpCheck(z2Real, z2Exp, t)
}

func TestMaskSuccess(t *testing.T) {
	var x = [][]float64{
		{-1, 6, 8},
		{4, 0, 8},
		{4, 6, -1},
	}
	zExp := [][]float64{
		{0, 1, 1},
		{1, 0, 1},
		{1, 1, 0},
	}
	xGPU := CopyH2D(x)
	zRealGPU := Mask(xGPU)
	CopyD2H(&zRealGPU)
	zReal := zRealGPU.CPU
	ExpCheck(zReal, zExp, t)
}

func TestAxpyESuccess(t *testing.T) {
	var x = [][]float64{
		{1, 1, 2},
		{4, 3, 1},
		{4, 1, 2},
	}
	b := 2.0
	c := 3.0
	zExp := [][]float64{
		{5, 5, 7},
		{11, 9, 5},
		{11, 5, 7},
	}
	xGPU := CopyH2D(x)
	zRealGPU := AxpyE(xGPU, b, c)
	CopyD2H(&zRealGPU)
	zReal := zRealGPU.CPU
	ExpCheck(zReal, zExp, t)
}

func TestExpSuccess(t *testing.T) {
	// exp(x + b) + c
	var x = [][]float64{
		{0, 0, 0},
		{0, 0, 0},
		{0, 0, 0},
	}
	zExp := [][]float64{
		{2, 2, 2},
		{2, 2, 2},
		{2, 2, 2},
	}
	b := 0.0
	c := 1.0
	xGPU := CopyH2D(x)
	zRealGPU := Exp(xGPU, b, c)
	CopyD2H(&zRealGPU)
	zReal := zRealGPU.CPU
	ExpCheck(zReal, zExp, t)
}

func TestExpTSuccess(t *testing.T) {
	// 1/ (exp(x + b) + c)
	var x = [][]float64{
		{0, 0, 0},
		{0, 0, 0},
		{0, 0, 0},
	}
	zExp := [][]float64{
		{0.5, 0.5, 0.5},
		{0.5, 0.5, 0.5},
		{0.5, 0.5, 0.5},
	}
	b := 0.0
	c := 1.0
	xGPU := CopyH2D(x)
	zRealGPU := ExpT(xGPU, b, c)
	CopyD2H(&zRealGPU)
	zReal := zRealGPU.CPU
	ExpCheck(zReal, zExp, t)
}

func TestlogSuccess(t *testing.T) {
	// 1/ (exp(x + b) + c)
	var x = [][]float64{
		{1, 1, 1},
		{1, 1, 1},
		{1, 1, 1},
	}
	zExp := [][]float64{
		{0, 0, 0},
		{0, 0, 0},
		{0, 0, 0},
	}
	b := 0.0
	xGPU := CopyH2D(x)
	zRealGPU := Log(xGPU, b)
	CopyD2H(&zRealGPU)
	zReal := zRealGPU.CPU
	ExpCheck(zReal, zExp, t)
}

func TestRandomNormSuccess(t *testing.T) {
	// make random matrix
	shape := []int{2,3}
	zRealGPU := RandomNorm(shape)
	CopyD2H(&zRealGPU)
	zReal := zRealGPU.CPU
	ExpRangeCheck(zReal, 0.0 , 1.0, t)

}

func TestTSuccess(t *testing.T) {
	var x = [][]float64{
		{2, 1, 3},
		{2, 2, 3},
	}
	zExp := [][]float64{
		{2, 2},
		{1, 2},
		{3, 3},
	}
	xGPU := CopyH2D(x)
	zRealGPU := T(xGPU)
	CopyD2H(&zRealGPU)
	zReal := zRealGPU.CPU
	ExpCheck(zReal, zExp, t)
}

func TestSubSuccess(t *testing.T) {
	var x = [][]float64{
		{2, 1, 1},
		{1, 1, 2},
		{1, 3, 1},
	}
	var y = [][]float64{
		{1, 1, 4},
		{2, 3, 1},
		{1, 1, 2},
	}
	zExp := [][]float64{
		{1, 0, -3},
		{-1, -2, 1},
		{0, 2, -1},
	}
	xGPU := CopyH2D(x)
	yGPU := CopyH2D(y)
	zRealGPU := Sub(xGPU, yGPU)
	CopyD2H(&zRealGPU)
	zReal := zRealGPU.CPU
	ExpCheck(zReal, zExp, t)
}

func TestMakeInitSuccess(t *testing.T) {
	zExp := [][]float64{
		{3, 3},
		{3, 3},
		{3, 3},
	}
	zRealGPU := MakeInit(3, 2, 3.0)
	CopyD2H(&zRealGPU)
	zReal := zRealGPU.CPU
	ExpCheck(zReal, zExp, t)
}

func TestSqrtTSuccess(t *testing.T) {
	// c[i] = 1 / (sqrtf(a[i] + b) + d);
	var x = [][]float64{
		{4, 4},
		{4, 4},
		{4, 4},
	}
	alpha := 0.0
	beta := 0.0
	zExp := [][]float64{
		{0.5, 0.5},
		{0.5, 0.5},
		{0.5, 0.5},
	}
	xGPU := CopyH2D(x)
	zRealGPU := SqrtT(xGPU, alpha, beta)
	CopyD2H(&zRealGPU)
	zReal := zRealGPU.CPU
	ExpCheck(zReal, zExp, t)
}

func TestArgMaxColSuccess(t *testing.T) {
	//
	var x = [][]float64{
		{1, 2},
		{3, 1},
		{4, 4},
	}
	zExp := [][]float64{
		{2, 2},
		{3, 3},
		{4, 4},
	}
	xGPU := CopyH2D(x)
	zRealGPU := ArgMaxCol(xGPU)
	CopyD2H(&zRealGPU)
	zReal := zRealGPU.CPU
	ExpCheck(zReal, zExp, t)
}

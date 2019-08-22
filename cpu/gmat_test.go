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
package cpu

import (
	"fmt"
	"testing"
)

func ExpCheck1D(zReal, zExp []float64, t *testing.T) {
	success := true
	for i, _ := range zExp {
		if zExp[i] != zReal[i] {
			success = false
		}
	}
	if !success {
		fmt.Println("Real:", zReal)
		fmt.Println("Exp:", zExp)
		t.Fatal("failed Test!")
	}
}

func ExpCheck(zReal, zExp [][]float64, t *testing.T) {
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

func ExpCheck4D(zReal [][][][]float64, zExp [][][][]float64, t *testing.T) {
	success := true
	for i, zx := range zExp {
		for j, zy := range zx {
			for k, zz := range zy {
				for l, _ := range zz {
					if zExp[i][j][k][l] != zReal[i][j][k][l] {
						success = false
					}
				}
			}
		}
	}
	if !success {
		fmt.Println("Real:", zReal)
		fmt.Println("Exp:", zExp)
		t.Fatal("failed Test!")
	}
}

var x = [][]float64{
	{1, 2, 3},
	{4, 5, 6},
	{7, 8, 9},
}
var y = [][]float64{
	{8, 7, 6},
	{5, 4, 3},
	{2, 1, 0},
}

func TestAddSuccess(t *testing.T) {
	zExp := [][]float64{
		{9, 9, 9},
		{9, 9, 9},
		{9, 9, 9},
	}
	zReal := Add(x, y)
	ExpCheck(zReal, zExp, t)
}

func minus(x float64) float64 {
	return x * -1
}

func TestApplySuccess(t *testing.T) {
	zExp := [][]float64{
		{-1, -2, -3},
		{-4, -5, -6},
		{-7, -8, -9},
	}
	zReal := Apply(x, minus)
	ExpCheck(zReal, zExp, t)
}

func TestSubSuccess(t *testing.T) {
	zExp := [][]float64{
		{-7, -5, -3},
		{-1, 1, 3},
		{5, 7, 9},
	}
	zReal := Sub(x, y)
	ExpCheck(zReal, zExp, t)
}

func TestTSuccess(t *testing.T) {
	zExp := [][]float64{
		{1, 4, 7},
		{2, 5, 8},
		{3, 6, 9},
	}
	zReal := T(x)
	ExpCheck(zReal, zExp, t)
}

func TestMulESuccess(t *testing.T) {
	zExp := [][]float64{
		{3, 6, 9},
		{12, 15, 18},
		{21, 24, 27},
	}
	zReal := MulE(x, 3)
	ExpCheck(zReal, zExp, t)
}

func TestMulSuccess(t *testing.T) {
	zExp := [][]float64{
		{8, 14, 18},
		{20, 20, 18},
		{14, 8, 0},
	}
	zReal := Mul(x, y)
	ExpCheck(zReal, zExp, t)
}

func TestDivSuccess(t *testing.T) {
	var xdiv = [][]float64{
		{4, 6, 8},
		{4, 6, 8},
		{4, 6, 8},
	}
	var ydiv = [][]float64{
		{2, 3, 4},
		{2, 3, 4},
		{2, 3, 4},
	}
	zExp := [][]float64{
		{2, 2, 2},
		{2, 2, 2},
		{2, 2, 2},
	}
	zReal := Div(xdiv, ydiv)
	ExpCheck(zReal, zExp, t)
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
	zReal := Dot(xdot, ydot)
	ExpCheck(zReal, zExp, t)
}

func TestSumRowSuccess(t *testing.T) {
	zExp := [][]float64{
		{12, 15, 18},
	}
	zReal := SumRow(x)
	ExpCheck(zReal, zExp, t)
}

func TestCastSuccess(t *testing.T) {
	z1 := [][]float64{
		{12, 15, 18},
	}
	z1Exp := [][]float64{
		{12, 15, 18},
		{12, 15, 18},
	}
	z1Real := Cast(z1, 2)
	ExpCheck(z1Real, z1Exp, t)
	z2 := [][]float64{
		{12},
		{15},
		{18},
	}
	z2Exp := [][]float64{
		{12, 12, 12},
		{15, 15, 15},
		{18, 18, 18},
	}
	z2Real := Cast(z2, 3)
	ExpCheck(z2Real, z2Exp, t)
}

func TestSumColSuccess(t *testing.T) {
	zExp := [][]float64{
		{6},
		{15},
		{24},
	}
	zReal := SumCol(x)
	ExpCheck(zReal, zExp, t)
}

func TestPad4DSuccess(t *testing.T) {
	var xpad = [][][][]float64{
		{
			{
				{4, 6, 8},
				{4, 6, 8},
				{4, 6, 8},
			},
			{
				{4, 6, 8},
				{4, 6, 8},
				{4, 6, 8},
			},
		},
	}

	var zExp = [][][][]float64{
		{
			{
				{0, 4, 6, 8, 0},
				{0, 4, 6, 8, 0},
				{0, 4, 6, 8, 0},
			},
			{
				{0, 4, 6, 8, 0},
				{0, 4, 6, 8, 0},
				{0, 4, 6, 8, 0},
			},
		},
	}
	var pad = [][]int{{0, 0}, {0, 0}, {0, 0}, {1, 1}}
	zReal := Pad4D(xpad, pad)
	ExpCheck4D(zReal, zExp, t)
}
func TestConv1DSuccess(t *testing.T) {
	var x2d = [][]float64{
		{10, 50, 60, 10, 20, 40, 30},
		{10, 50, 60, 10, 20, 40, 30}}
	var y2d = [][]float64{{2, 3, 4}, {2, 3, 4}}
	var zExp = [][]float64{
		{230, 410, 320, 230, 240, 280, 170},
		{230, 410, 320, 230, 240, 280, 170},
	}
	zReal := Conv1D(x2d, y2d, 1)
	ExpCheck(zReal, zExp, t)
}

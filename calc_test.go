package gmat

import (
	"testing"
	"fmt"
)


func ExpCheck(zReal [][]float64, zExp [][]float64, t *testing.T) {
	success := true
	for i , zArray := range zExp {
		for j, _ := range zArray {
			if zExp[i][j] != zReal[i][j] {
					success = false
			}
		}
	}
	if !success {
		fmt.Println("Real:",zReal)
		fmt.Println("Exp:",zExp)
		t.Fatal("failed Test!")
	}
}

var x =[][]float64{
	{1,2,3},
	{4,5,6},
	{7,8,9},
}
var y =[][]float64{
	{8,7,6},
	{5,4,3},
	{2,1,0},
}

func TestAddSuccess(t *testing.T) {
	zExp := [][]float64{
		{9,9,9},
		{9,9,9},
		{9,9,9},
	}
	zReal := Add(x,y)
	ExpCheck(zReal, zExp, t)
}

func TestSubSuccess(t *testing.T) {
	zExp := [][]float64{
		{-7,-5,-3},
		{-1,1,3},
		{5,7,9},
	}
	zReal := Sub(x,y)
	ExpCheck(zReal, zExp, t)
}

func TestMulESuccess(t *testing.T) {
	zExp := [][]float64{
		{3,6,9},
		{12,15,18},
		{21,24,27},
	}
	zReal := MulE(x,3)
	ExpCheck(zReal, zExp, t)
}

func TestMulSuccess(t *testing.T) {
	zExp := [][]float64{
		{8,14,18},
		{20,20,18},
		{14,8,0},
	}
	zReal := Mul(x,y)
	ExpCheck(zReal, zExp, t)
}

func TestDotSuccess(t *testing.T) {
	xdot :=[][]float64{
		{1,2,3},
		{4,5,6},
		{7,8,9},
	}
	ydot :=[][]float64{
		{8,7},
		{5,4},
		{2,1},
	}
	zExp := [][]float64{
		{24,18},
		{69,54},
		{114,90},
	}
	zReal := Dot(xdot, ydot)
	ExpCheck(zReal, zExp, t)
}

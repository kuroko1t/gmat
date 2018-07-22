package data

import "C"

type Data struct {
	CPU   [][]float64
	CPU4D [][][][]float64
	GPU   *C.float
	Shape []int
}

// +build cpu

package gmat

import (
	"github.com/kuroko1t/gmat/cpu"
	"log"
)

type Tensor cpu.Tensor

func Make2D(n int, m int) Tensor {
	z := Tensor{Shape: []int{n, m}}
	z.CPU = cpu.Make2D(n, m)
	return z
}

func Make2DInitArray(x [][]float64) Tensor {
	z := Tensor{CPU: x}
	z.CPU = x
	return z
}

func Make4D(n int, c int, h int, w int) [][][][]float64 {
	z := make([][][][]float64, n)
	for i := range z {
		z[i] = make([][][]float64, c)
		for j := range z[i] {
			z[i][j] = make([][]float64, h)
			for k := range z[i][j] {
				z[i][j][k] = make([]float64, w)
			}
		}
	}
	return z
}

func Make6D(n int, c int, h int, w int, x int, y int) [][][][][][]float64 {
	z := make([][][][][][]float64, n)
	for i := range z {
		z[i] = make([][][][][]float64, c)
		for j := range z[i] {
			z[i][j] = make([][][][]float64, h)
			for k := range z[i][j] {
				z[i][j][k] = make([][][]float64, w)
				for l := range z[i][j][k] {
					z[i][j][k][l] = make([][]float64, x)
					for m := range z[i][j][k][l] {
						z[i][j][k][l][m] = make([]float64, y)
					}
				}
			}
		}
	}
	return z
}

func Trans2D(input Tensor, n int, c int) Tensor {
	z := Tensor{}
	z.CPU = cpu.Trans2D(input.CPU, n, c)
	return z
}

//func Trans4D(input [][][][]float64, n int, c int, h int, w int) [][][][]float64 {
// 	if n >= 4 || c >= 4 || h >= 4 || w >= 4 {
// 		log.Fatal("need to set 4 below param")
// 	}
// 	inN := len(input)
// 	inC := len(input[0])
// 	inH := len(input[0][0])
// 	inW := len(input[0][0][0])
// 	tranmap := map[int]int{0: inN, 1: inC, 2: inH, 3: inW}
// 	z := Make4D(tranmap[n], tranmap[c], tranmap[h], tranmap[w])
// 	for i := range z {
// 		for j := range z[i] {
// 			for k := range z[i][j] {
// 				for l := range z[i][j][k] {
// 					var amap = map[int]int{0: i, 1: j, 2: k, 3: l}
// 					z[i][j][k][l] = input[amap[n]][amap[c]][amap[h]][amap[w]]
// 				}
// 			}
// 		}
// 	}
// 	return z
//}

//func Trans6D(input [][][][][][]float64, n int, c int, h int, w int, x int, y int) [][][][][][]float64 {
// 	if n >= 6 || c >= 6 || h >= 6 || w >= 6 || x >= 6 || y >= 6 {
// 		log.Fatal("need to set 6 below param")
// 	}
// 	inN := len(input)
// 	inC := len(input[0])
// 	inH := len(input[0][0])
// 	inW := len(input[0][0][0])
// 	inX := len(input[0][0][0][0])
// 	inY := len(input[0][0][0][0][0])
// 	tranmap := map[int]int{0: inN, 1: inC, 2: inH, 3: inW, 4: inX, 5: inY}
// 	z := Make6D(tranmap[n], tranmap[c], tranmap[h], tranmap[w], tranmap[x], tranmap[y])
// 	for i := range z {
// 		for j := range z[i] {
// 			for k := range z[i][j] {
// 				for l := range z[i][j][k] {
// 					for m := range z[i][j][k][l] {
// 						for o := range z[i][j][k][l][m] {
// 							var amap = map[int]int{0: i, 1: j, 2: k, 3: l, 4: m, 5: o}
// 							z[i][j][k][l][m][o] = input[amap[n]][amap[c]][amap[h]][amap[w]][amap[x]][amap[y]]
// 						}
// 					}
// 				}
// 			}
// 		}
// 	}
// 	return z
//}

//func Reshape2D(input [][]float64, reN int, reC int, reH int, reW int) [][][][]float64 {
// 	n := len(input)
// 	c := len(input[0])
// 	if reW == -1 {
// 		reW = n * c / (reN * reC * reH)
// 	}
// 	var input1D []float64
// 	tmp := 0
// 	for i := range input {
// 		for j := range input[i] {
// 			input1D[tmp] = input[i][j]
// 			tmp++
// 		}
// 	}
// 	result := Make4D(reN, reC, reH, reW)
// 	tmp = 0
// 	for i := range result {
// 		for j := range result[i] {
// 			for k := range result[i][j] {
// 				for l := range result[i][j][k] {
// 					result[i][j][k][l] = input1D[tmp]
// 					tmp++
// 				}
// 			}
// 		}
// 	}
// 	return result
//}

//func Reshape2D2D(input [][]float64, reX int, reY int) [][]float64 {
// 	n := len(input)
// 	c := len(input[0])
// 	if reX == -1 {
// 		reX = n * c / (reY)
// 	} else if reY == -1 {
// 		reY = n * c / (reX)
// 	}
// 	var input1D []float64
// 	tmp := 0
// 	for i := range input {
// 		for j := range input[i] {
// 			input1D[tmp] = input[i][j]
// 			tmp++
// 		}
// 	}
// 	result := Make2D(reX, reY)
// 	tmp = 0
// 	for i := range result {
// 		for j := range result[i] {
// 			result[i][j] = input1D[tmp]
// 			tmp++
// 		}
// 	}
// 	return result
//}

func Reshape2D6D(input [][]float64, reN int, reC int, reH int, reW int, reX int, reY int) [][][][][][]float64 {
	var input1D []float64
	tmp := 0
	for i := range input {
		for j := range input[i] {
			input1D[tmp] = input[i][j]
			tmp++
		}
	}
	result := Make6D(reN, reC, reH, reW, reX, reY)
	tmp = 0
	for i := range result {
		for j := range result[i] {
			for k := range result[i][j] {
				for l := range result[i][j][k] {
					for m := range result[i][j][k][l] {
						for n := range result[i][j][k][l][m] {
							result[i][j][k][l][m][n] = input1D[tmp]
							tmp++
						}
					}
				}
			}
		}
	}
	return result
}

func Reshape4D(input Tensor, reX int, reY int) Tensor {
	z := Tensor{}
	z.CPU = cpu.Reshape4D(input.CPU4D, reX, reY)
	return z
}

func Reshape4D6D(input [][][][]float64, reN int, reC int, reH int, reW int, reX int, reY int) [][][][][][]float64 {
	//n, c, h, w := Shape4D(input)

	var input1D []float64
	tmp := 0
	for i := range input {
		for j := range input[i] {
			for k := range input[i][j] {
				for l := range input[i][j][k] {
					input1D[tmp] = input[i][j][k][l]
					tmp++
				}
			}
		}
	}
	result := Make6D(reN, reC, reH, reW, reX, reY)
	tmp = 0
	for i := range result {
		for j := range result[i] {
			for k := range result[i][j] {
				for l := range result[i][j][k] {
					for m := range result[i][j][k][l] {
						for n := range result[i][j][k][l][m] {
							result[i][j][k][l][m][n] = input1D[tmp]
							tmp++
						}
					}
				}
			}
		}
	}
	return result
}

//func Reshape6D(input [][][][][][]float64, reX int, reY int) [][]float64 {
// 	n := len(input)
// 	c := len(input[0])
// 	h := len(input[0][0])
// 	w := len(input[0][0][0])
// 	x := len(input[0][0][0][0])
// 	y := len(input[0][0][0][0][0])
// 	if reY == -1 {
// 		reY = n * c * h * w * x * y / reX
// 	}
// 	var input1D []float64
// 	tmp := 0
// 	for i := range input {
// 		for j := range input[i] {
// 			for k := range input[i][j] {
// 				for l := range input[i][j][k] {
// 					for m := range input[i][j][l][l] {
// 						for o := range input[i][j][k][l][m] {
// 							input1D[tmp] = input[i][j][k][l][m][o]
// 							tmp++
// 						}
// 					}
// 				}
// 			}
// 		}
// 	}
// 	result := Make2D(reX, reY)
// 	tmp = 0
// 	for i := range result {
// 		for j := range result[i] {
// 			result[i][j] = input1D[tmp]
// 			tmp++
// 		}
// 	}
// 	return result
//}

func Shape2D(x Tensor) (n int, c int) {
	n, c = cpu.Shape2D(x.CPU)
	return n, c
}

func Shape4D(input Tensor) (n int, c int, h int, w int) {
	n, c, h, w = cpu.Shape4D(input.CPU4D)
	return n, c, h, w
}

func Shape6D(input [][][][][][]float64) (n int, c int, h int, w int, x int, y int) {
	n = len(input)
	c = len(input[0])
	h = len(input[0][0])
	w = len(input[0][0][0])
	x = len(input[0][0][0][0])
	y = len(input[0][0][0][0][0])
	return n, c, h, w, x, y
}

func Pad4D(input [][][][]float64, pad [][]int) [][][][]float64 {
	padN := len(pad)
	padM := len(pad[0])
	n := len(input)
	c := len(input[0])
	h := len(input[0][0])
	w := len(input[0][0][0])
	if padN != 4 && padM == 2 {
		log.Fatal("incorrect padding dim!!")
	}
	zN := n + pad[0][0] + pad[0][1]
	zC := c + pad[1][0] + pad[1][1]
	zH := h + pad[2][0] + pad[2][1]
	zW := w + pad[3][0] + pad[3][1]
	z := Make4D(zN, zC, zH, zW)
	for i := range z {
		for j := range z[i] {
			for k := range z[i][j] {
				for l := range z[i][j][k] {
					if (pad[0][0] <= i) && (pad[1][0] <= j) &&
						(pad[2][0] <= k) && (pad[3][0] <= l) &&
						(n+pad[0][1]-1 >= i) && (c+pad[1][1]-1 >= j) &&
						(h+pad[2][1]-1 >= k) && (w+pad[3][1]-1 >= l) {
						z[i][j][k][l] = input[i-pad[0][0]][j-pad[1][0]][k-pad[2][0]][l-pad[3][0]]
					}
				}
			}
		}
	}
	return z
}

func MakeInit(n int, m int, value float64) Tensor {
	z := Tensor{}
	z.CPU = cpu.MakeInit(n, m, value)
	return z
}

func Add(x, y Tensor) Tensor {
	z := Tensor{}
	z.CPU = cpu.Add(x.CPU, y.CPU)
	return z
}

func AddE(x Tensor, y float64) Tensor {
	z := Tensor{}
	z.CPU = cpu.AddE(x.CPU, y)
	return z
}

func Sub(x, y Tensor) Tensor {
	z := x
	z.CPU = cpu.Sub(x.CPU, y.CPU)
	return z
}

func SubE(x Tensor, y float64) Tensor {
	z := Tensor{}
	z.CPU = cpu.SubE(x.CPU, y)
	return z
}

func MulE(x Tensor, y float64) Tensor {
	z := Tensor{}
	z.CPU = cpu.MulE(x.CPU, y)
	return z
}

func Mul(x, y Tensor) Tensor {
	z := Tensor{}
	z.CPU = cpu.Mul(x.CPU, y.CPU)
	return z
}

func Div(x, y Tensor) Tensor {
	z := Tensor{}
	z.CPU = cpu.Div(x.CPU, y.CPU)
	return z
}

func T(x Tensor) Tensor {
	z := Tensor{}
	z.CPU = cpu.T(x.CPU)
	return z
}

func Apply(x Tensor, fn func(float64) float64) Tensor {
	z := Tensor{}
	z.CPU = cpu.Apply(x.CPU, fn)
	return z
}

func Dot(x, y Tensor) Tensor {
	z := Tensor{}
	z.CPU = cpu.Dot(x.CPU, y.CPU)
	return z
}

func SumRow(x Tensor) Tensor {
	//sum | direction [a,b]
	//    ^           [a,b]
	z := Tensor{}
	z.CPU = cpu.SumRow(x.CPU)
	return z
}

func SumCol(x Tensor) Tensor {
	//sum -> direction [a,a]
	//				   [b,b]
	z := Tensor{}
	z.CPU = cpu.SumCol(x.CPU)
	return z
}

func MaxCol(x Tensor) Tensor {
	//sum -> direction [a,a]
	//				   [b,b]
	z := Tensor{}
	z.CPU = cpu.MaxCol(x.CPU)
	return z
}

func ArgMaxCol(x Tensor) [][]int {
	//sum -> direction [a,a]
	//				   [b,b]
	//z := Tensor{}
	maxArray := cpu.ArgMaxCol(x.CPU)
	return maxArray
}

func RandomNorm2D(r int, c int, init float64) Tensor {
	z := Tensor{}
	z.CPU = cpu.RandomNorm2D(r, c, init)
	return z
}

func HeNorm2D(r int, c int) Tensor {
	z := Tensor{}
	z.CPU = cpu.HeNorm2D(r, c)
	return z
}

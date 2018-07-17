package gmat

import (
	"log"
	"math/rand"
)

func Make(n int, m int) [][]float64 {
	z := make([][]float64, n)
	for i := 0; i < n; i++ {
		z[i] = make([]float64, m)
	}
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

func Trans2D(input [][]float64, n int, c int) [][]float64 {
	if n >= 2 || c >= 2 {
		log.Fatal("need to set 2 below param")
	}
	inN := len(input)
	inC := len(input[0])
	tranmap := map[int]int{0: inN, 1: inC}
	z := Make(tranmap[n], tranmap[c])
	for i := range z {
		for j := range z[i] {
			var amap = map[int]int{0: i, 1: j}
			z[i][j] = input[amap[n]][amap[c]]
		}
	}
	return z
}

func Trans4D(input [][][][]float64, n int, c int, h int, w int) [][][][]float64 {
	if n >= 4 || c >= 4 || h >= 4 || w >= 4 {
		log.Fatal("need to set 4 below param")
	}
	inN := len(input)
	inC := len(input[0])
	inH := len(input[0][0])
	inW := len(input[0][0][0])
	tranmap := map[int]int{0: inN, 1: inC, 2: inH, 3: inW}
	z := Make4D(tranmap[n], tranmap[c], tranmap[h], tranmap[w])
	for i := range z {
		for j := range z[i] {
			for k := range z[i][j] {
				for l := range z[i][j][k] {
					var amap = map[int]int{0: i, 1: j, 2: k, 3: l}
					z[i][j][k][l] = input[amap[n]][amap[c]][amap[h]][amap[w]]
				}
			}
		}
	}
	return z
}

func Trans6D(input [][][][][][]float64, n int, c int, h int, w int, x int, y int) [][][][][][]float64 {
	if n >= 6 || c >= 6 || h >= 6 || w >= 6 || x >= 6 || y >= 6 {
		log.Fatal("need to set 6 below param")
	}
	inN := len(input)
	inC := len(input[0])
	inH := len(input[0][0])
	inW := len(input[0][0][0])
	inX := len(input[0][0][0][0])
	inY := len(input[0][0][0][0][0])
	tranmap := map[int]int{0: inN, 1: inC, 2: inH, 3: inW, 4: inX, 5: inY}
	z := Make6D(tranmap[n], tranmap[c], tranmap[h], tranmap[w], tranmap[x], tranmap[y])
	for i := range z {
		for j := range z[i] {
			for k := range z[i][j] {
				for l := range z[i][j][k] {
					for m := range z[i][j][k][l] {
						for o := range z[i][j][k][l][m] {
							var amap = map[int]int{0: i, 1: j, 2: k, 3: l, 4: m, 5: o}
							z[i][j][k][l][m][o] = input[amap[n]][amap[c]][amap[h]][amap[w]][amap[x]][amap[y]]
						}
					}
				}
			}
		}
	}
	return z
}

func Reshape2D(input [][]float64, reN int, reC int, reH int, reW int) [][][][]float64 {
	n := len(input)
	c := len(input[0])
	if reW == -1 {
		reW = n * c / (reN * reC * reH)
	}
	var input1D []float64
	tmp := 0
	for i := range input {
		for j := range input[i] {
			input1D[tmp] = input[i][j]
			tmp++
		}
	}
	result := Make4D(reN, reC, reH, reW)
	tmp = 0
	for i := range result {
		for j := range result[i] {
			for k := range result[i][j] {
				for l := range result[i][j][k] {
					result[i][j][k][l] = input1D[tmp]
					tmp++
				}
			}
		}
	}
	return result
}

func Reshape2D2D(input [][]float64, reX int, reY int) [][]float64 {
	n := len(input)
	c := len(input[0])
	if reX == -1 {
		reX = n * c / (reY)
	} else if reY == -1 {
		reY = n * c / (reX)
	}
	var input1D []float64
	tmp := 0
	for i := range input {
		for j := range input[i] {
			input1D[tmp] = input[i][j]
			tmp++
		}
	}
	result := Make(reX, reY)
	tmp = 0
	for i := range result {
		for j := range result[i] {
			result[i][j] = input1D[tmp]
			tmp++
		}
	}
	return result
}

func Reshape2D6D(input [][]float64, reN int, reC int, reH int, reW int, reX int, reY int) [][][][][][]float64 {
	//n := len(input)
	//c := len(input[0])
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

func Reshape4D(input [][][][]float64, reX int, reY int) [][]float64 {
	n := len(input)
	c := len(input[0])
	h := len(input[0][0])
	w := len(input[0][0][0])
	if reY == -1 {
		reY = n * c * h * w / reX
	} else if reX == -1 {
		reX = n * c * h * w / reY
	}
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
	result := Make(reX, reY)
	tmp = 0
	for i := range result {
		for j := range result[i] {
			result[i][j] = input1D[tmp]
			tmp++
		}
	}
	return result
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

func Reshape6D(input [][][][][][]float64, reX int, reY int) [][]float64 {
	n := len(input)
	c := len(input[0])
	h := len(input[0][0])
	w := len(input[0][0][0])
	x := len(input[0][0][0][0])
	y := len(input[0][0][0][0][0])
	if reY == -1 {
		reY = n * c * h * w * x * y / reX
	}
	var input1D []float64
	tmp := 0
	for i := range input {
		for j := range input[i] {
			for k := range input[i][j] {
				for l := range input[i][j][k] {
					for m := range input[i][j][l][l] {
						for o := range input[i][j][k][l][m] {
							input1D[tmp] = input[i][j][k][l][m][o]
							tmp++
						}
					}
				}
			}
		}
	}
	result := Make(reX, reY)
	tmp = 0
	for i := range result {
		for j := range result[i] {
			result[i][j] = input1D[tmp]
			tmp++
		}
	}
	return result
}

func Shape2D(input [][]float64) (n int, c int) {
	n = len(input)
	c = len(input[0])
	return n, c
}

func Shape4D(input [][][][]float64) (n int, c int, h int, w int) {
	n = len(input)
	c = len(input[0])
	h = len(input[0][0])
	w = len(input[0][0][0])
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

func MakeInit(n int, m int, value float64) [][]float64 {
	z := make([][]float64, n)
	for i := 0; i < n; i++ {
		z[i] = make([]float64, m)
	}
	for i, zArray := range z {
		for j, _ := range zArray {
			z[i][j] = value
		}
	}
	return z
}

func Add(x [][]float64, y [][]float64) [][]float64 {
	n := len(x)
	m := len(x[0])
	z := make([][]float64, n)
	for i := 0; i < n; i++ {
		z[i] = make([]float64, m)

	}
	for i, xArray := range x {
		for j, _ := range xArray {
			z[i][j] = x[i][j] + y[i][j]
		}
	}
	return z
}

func AddE(x [][]float64, y float64) [][]float64 {
	n := len(x)
	m := len(x[0])
	z := make([][]float64, n)
	for i := 0; i < n; i++ {
		z[i] = make([]float64, m)

	}
	for i, xArray := range x {
		for j, _ := range xArray {
			z[i][j] = x[i][j] + y
		}
	}
	return z
}

func Sub(x [][]float64, y [][]float64) [][]float64 {
	n := len(x)
	m := len(x[0])
	z := make([][]float64, n)
	for i := 0; i < n; i++ {
		z[i] = make([]float64, m)

	}
	for i, xArray := range x {
		for j, _ := range xArray {
			z[i][j] = x[i][j] - y[i][j]
		}
	}
	return z
}

func SubE(x [][]float64, y float64) [][]float64 {
	n := len(x)
	m := len(x[0])
	z := make([][]float64, n)
	for i := 0; i < n; i++ {
		z[i] = make([]float64, m)

	}
	for i, xArray := range x {
		for j, _ := range xArray {
			z[i][j] = x[i][j] - y
		}
	}
	return z
}

func MulE(x [][]float64, y float64) [][]float64 {
	n := len(x)
	m := len(x[0])
	z := make([][]float64, n)
	for i := 0; i < n; i++ {
		z[i] = make([]float64, m)
	}
	for i, xArray := range x {
		for j, _ := range xArray {
			z[i][j] = x[i][j] * y
		}
	}
	return z
}

func Mul(x [][]float64, y [][]float64) [][]float64 {
	n := len(x)
	m := len(x[0])
	z := make([][]float64, n)
	for i := 0; i < n; i++ {
		z[i] = make([]float64, m)
	}
	for i, xArray := range x {
		for j, _ := range xArray {
			z[i][j] = x[i][j] * y[i][j]
		}
	}
	return z
}

func Div(x [][]float64, y [][]float64) [][]float64 {
	n := len(x)
	m := len(x[0])
	z := make([][]float64, n)
	for i := 0; i < n; i++ {
		z[i] = make([]float64, m)
	}
	for i, xArray := range x {
		for j, _ := range xArray {
			z[i][j] = x[i][j] / y[i][j]
		}
	}
	return z
}

func T(x [][]float64) [][]float64 {
	n := len(x)
	m := len(x[0])
	z := make([][]float64, m)
	for i := 0; i < m; i++ {
		z[i] = make([]float64, n)
	}
	for i, zArray := range z {
		for j, _ := range zArray {
			z[i][j] = x[j][i]
		}
	}
	return z
}

func Apply(x [][]float64, fn func(float64) float64) [][]float64 {
	n := len(x)
	m := len(x[0])
	z := make([][]float64, n)
	for i := 0; i < n; i++ {
		z[i] = make([]float64, m)
	}
	for i, xArray := range x {
		for j, _ := range xArray {
			z[i][j] = fn(x[i][j])
		}
	}
	return z
}

func Dot(x [][]float64, y [][]float64) [][]float64 {
	nx := len(x)
	mx := len(x[0])
	ny := len(y)
	my := len(y[0])
	if mx != ny {
		log.Fatal("mismatch matrix number")
	}
	z := make([][]float64, nx)
	for i := 0; i < nx; i++ {
		z[i] = make([]float64, my)

	}
	zElem := 0.0
	for zcol := 0; zcol < nx; zcol++ {
		for zrow := 0; zrow < my; zrow++ {
			for i := 0; i < mx; i++ {
				zElem += x[zcol][i] * y[i][zrow]
			}
			z[zcol][zrow] = zElem
			zElem = 0
		}
	}
	return z
}

func SumRow(x [][]float64) [][]float64 {
	//sum | direction [a,b]
	//    ^           [a,b]
	n := len(x)
	m := len(x[0])
	sumArray := make([][]float64, n)
	for i := 0; i < n; i++ {
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

func SumCol(x [][]float64) [][]float64 {
	//sum -> direction [a,a]
	//				   [b,b]
	n := len(x)
	m := len(x[0])
	sumArray := make([][]float64, n)
	for i := 0; i < n; i++ {
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

func MaxCol(x [][]float64) [][]float64 {
	//sum -> direction [a,a]
	//				   [b,b]
	n := len(x)
	m := len(x[0])
	maxArray := make([][]float64, n)
	for i := 0; i < n; i++ {
		maxArray[i] = make([]float64, m)
	}
	for j := 0; j < n; j++ {
		max := 0.0
		for i := 0; i < m; i++ {
			if x[j][i] > max {
				max = x[j][i]
			}
		}
		for i := 0; i < m; i++ {
			maxArray[j][i] = max
		}
	}
	return maxArray
}

func ArgMaxCol(x [][]float64) [][]int {
	//sum -> direction [a,a]
	//				   [b,b]
	n := len(x)
	m := len(x[0])
	maxArray := make([][]int, n)
	for i := 0; i < n; i++ {
		maxArray[i] = make([]int, m)
	}
	index := 0
	for j := 0; j < n; j++ {
		max := 0.0
		for i := 0; i < m; i++ {
			if x[j][i] > max {
				max = x[j][i]
				index = i
			}
		}
		for i := 0; i < m; i++ {
			maxArray[j][i] = index
		}
	}
	return maxArray
}

func RandomNorm2D(r int, c int, init float64) [][]float64 {
	z := make([][]float64, r)
	for i := 0; i < r; i++ {
		z[i] = make([]float64, c)
	}
	for i, zArray := range z {
		for j, _ := range zArray {
			z[i][j] = rand.NormFloat64() * init
		}
	}
	return z
}

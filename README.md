# gmat
golang matrix library

# Sample

```golang
package main

import (
	"github.com/kuroko1t/gmat"
	"fmt"
)


func main() {
    xdot := [][]float64{
        {2, 3, 6},
        {1, 4, 4},
        {2, 1, 5},
        {3, 4, 3},
    }
    ydot := [][]float64{
        {3, 1, 2, 2, 3},
        {2, 4, 2, 4, 6},
        {3, 4, 6, 3, 8},
    }
    xtensor := gmat.Make2DInitArray(xdot)
    ytensor := gmat.Make2DInitArray(ydot)
    ztensor := gmat.Dot(xtensor, ytensor)
    fmt.Println(ztensor.CPU)
}
```

# API
* Make(shape []int) Tensor
* Make2DInitArray(x [][]float64) Tensor
* MakeInit(n int, m int, value float64) Tensor
* Add(x, y Tensor) Tensor
* AddE(x Tensor, y float64) Tensor
* Sub(x, y Tensor) Tensor
* SubE(x Tensor, y float64) Tensor
* MulE(x Tensor, y float64) Tensor
* Mul(x, y Tensor) Tensor
* Div(x, y Tensor) Tensor
* T(x Tensor) Tensor
* Apply(x Tensor, fn func(float64) float64) Tensor
* Dot(x, y Tensor) Tensor
* SumRow(x Tensor) Tensor
* SumCol(x Tensor) Tensor
* Cast(x Tensor, castSize int) Tensor
* MaxCol(x Tensor) Tensor
* ArgMaxCol(x Tensor) [][]int
* RandomNorm2D(r int, c int, init float64) Tensor
* HeNorm2D(r int, c int) Tensor
* Conv1D(x, filter Tensor, stride int) Tensor

# License

gdeep is licensed under the Apache License, Version2.0

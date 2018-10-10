
__global__
void kgmul(float *a, float *b, float *c)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  c[i] = a[i] * b[i];
}

__global__
void kgmule(float *a, float b, float *c)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  c[i] = a[i] * b;
}

__global__
void kgdiv(float *a, float *b, float *c)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  c[i] = a[i] / b[i];
}

extern "C" {
  void gmul(int blocks, int threads, float *a, float *b, float *c) {
    kgmul<<<blocks, threads>>>(a, b, c);
  }
  void gmule(int blocks, int threads, float *a, float b, float *c) {
    kgmule<<<blocks, threads>>>(a, b, c);
  }

  void gdiv(int blocks, int threads, float *a, float *b, float *c) {
    kgdiv<<<blocks, threads>>>(a, b, c);
  }
}

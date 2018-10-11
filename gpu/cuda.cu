
__global__
void kgmul(float *a, float *b, float *c) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  c[i] = a[i] * b[i];
}

__global__
void kgmule(float *a, float b, float *c) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  c[i] = a[i] * b;
}

__global__
void kgdiv(float *a, float *b, float *c) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  c[i] = a[i] / b[i];
}

__global__
void kmasc(float *a, float *c) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (a[i] <= 0) {
    c[i] = 0;
  } else {
    c[i] = 1;
  }
}

__global__
void kaxpye(float *a, float b, float c, float *d) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  d[i] = a[i] *b + c;
}

__global__
void kexp(float *a, float b, float c, float *d) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  d[i] = expf(a[i] * b) + c;
}

__global__
void kexpT(float *a, float b, float c, float *d) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  d[i] = 1/ (expf(a[i] * b) + c);
}


__global__
void klog(float *a, float b, float *c) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  c[i] = logf(a[i] + b);
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
  void gmasc(int blocks, int threads, float *a, float *c) {
    kmasc<<<blocks, threads>>>(a, c);
  }
  void gaxpye(int blocks, int threads, float *a, float b, float c, float *d) {
    kaxpye<<<blocks, threads>>>(a, b, c, d);
  }
  void gexp(int blocks, int threads, float *a, float b, float c, float *d) {
    kexp<<<blocks, threads>>>(a, b, c, d);
  }
  void gexpT(int blocks, int threads, float *a, float b, float c, float *d) {
    kexpT<<<blocks, threads>>>(a, b, c, d);
  }
  void glog(int blocks, int threads, float *a, float b, float *c) {
    klog<<<blocks, threads>>>(a, b, c);
  }
}

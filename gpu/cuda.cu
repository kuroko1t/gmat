#include <thrust/device_vector.h>
#include <thrust/fill.h>

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
void ksub(float *a, float *b, float *c) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  c[i] = a[i] - b[i];
}

__global__
void kfill(float *a, float b) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  a[i] = b;
}

__global__
void kmask(float *a, float *c) {
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

__global__
void ksqrtT(float *a, float b, float d, float *c) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  c[i] = 1 / (sqrtf(a[i] + b) + d);
}

__global__
void kdeviceMemset(float *a, float *c) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    c[i] = a[0];
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
  void gsub(int blocks, int threads, float *a, float *b, float *c) {
    ksub<<<blocks, threads>>>(a, b, c);
  }
  void gmask(int blocks, int threads, float *a, float *c) {
    kmask<<<blocks, threads>>>(a, c);
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
  void gfill(int blocks, int threads, float *a, float b) {
    kfill<<<blocks, threads>>>(a, b);
  }
  void glog(int blocks, int threads, float *a, float b, float *c) {
    klog<<<blocks, threads>>>(a, b, c);
  }
  void gsqrtT(int blocks, int threads, float *a, float b, float d, float *c) {
    ksqrtT<<<blocks, threads>>>(a, b, d, c);
  }
  void gdeviceMemset(int blocks, int threads, float *a, float *c) {
    kdeviceMemset<<<blocks, threads>>>(a, c);
  }
  //float* gdev_memset(size_t N, float value) {
  //  thrust::device_ptr<float> dev_ptr = thrust::device_malloc<float>(N);
  //  float* raw_ptr = thrust::raw_pointer_cast(dev_ptr);
  //  return raw_ptr;
  //}
}

extern "C++"{

}

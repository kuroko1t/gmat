void gmul(int blocks, int threads, float *a, float *b, float *c);
void gmule(int blocks, int threads, float *a, float b, float *c);
void gdiv(int blocks, int threads, float *a, float *b, float *c);
void gmask(int blocks, int threads, float *a, float *c);
void gaxpye(int blocks, int threads, float *a, float b, float c, float *d);
void gexp(int blocks, int threads, float *a, float b, float c, float *d);
void gexpT(int blocks, int threads, float *a, float b, float c, float *d);
void glog(int blocks, int threads, float *a, float b, float *c);

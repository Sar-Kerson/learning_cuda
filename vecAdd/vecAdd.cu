#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
void sumOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; ++idx) {
        C[idx] = A[idx] + B[idx];
    }
}

void initData(float *ip, int size)
{
    time_t t;
    srand((unsigned int) time(&t));

    for (int i = 0; i < size; ++i) {
        ip[i] = (float)(rand() & 0xff) / 10.0f;
    }
}

__global__ void sumOnDevice(float *A, float *B, float *C)
{
    int id = threadIdx.x;
    C[id] = A[id] + B[id];
}

int main(void)
{
    int nElem = 1024;
    size_t nByte = nElem * sizeof(float);
    float *d_A, *d_B, *d_C;
    float *h_A, *h_B, *h_C, *res;
    h_A = (float *)malloc(nByte);
    h_B = (float *)malloc(nByte);
    h_C = (float *)malloc(nByte);
    res = (float *)malloc(nByte);

    initData(h_A, nElem);
    initData(h_B, nElem);

    cudaMalloc((float**)&d_A, nByte);
    cudaMalloc((float**)&d_B, nByte);
    cudaMalloc((float**)&d_C, nByte);

    cudaMemcpy(d_A, h_A, nByte, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nByte, cudaMemcpyHostToDevice);

    sumOnDevice <<<1, nElem>>>(d_A, d_B, d_C);
    //cudaDeviceReset();
    cudaDeviceSynchronize();
    sumOnHost(h_A, h_B, h_C, nElem);

    cudaMemcpy(res, d_C, nByte, cudaMemcpyDeviceToHost);
    
    printf("%f\t%f\n", h_C[0], h_C[1000]);
    printf("%f\t%f\n", res[0], res[1000]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

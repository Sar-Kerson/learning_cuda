#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define CHECK(call)                                                    \
{                                                                      \
    const cudaError_t error = call;                                    \
    if (error != cudaSuccess)                                          \
    {                                                                  \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                  \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));\
        exit(1);                                                       \
    }                                                                  \
}


double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


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

__global__ void sumOnDevice_single_block(float *A, float *B, float *C)
{
    int id = threadIdx.x;
    C[id] = A[id] + B[id];
    //printf("blockDim.x: %d\t", blockDim.x);
    //printf("blockDim.y: %d\t", blockDim.y);
    //printf("blockDim.z: %d\n", blockDim.z);
}

__global__ void sumOnDevice_multi_block(float *A, float *B, float *C)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    C[id] = A[id] + B[id];
}

int main(void)
{
    int nElem = 102400;   //2048 will fail
    size_t nByte = nElem * sizeof(float);
    float *d_A, *d_B, *d_C;
    float *h_A, *h_B, *h_C, *res;
    h_A = (float *)malloc(nByte);
    h_B = (float *)malloc(nByte);
    h_C = (float *)malloc(nByte);
    res = (float *)malloc(nByte);

    initData(h_A, nElem);
    initData(h_B, nElem);

    CHECK(cudaMalloc((float**)&d_A, nByte));
    CHECK(cudaMalloc((float**)&d_B, nByte));
    CHECK(cudaMalloc((float**)&d_C, nByte));

    CHECK(cudaMemcpy(d_A, h_A, nByte, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nByte, cudaMemcpyHostToDevice));

    double iStart = cpuSecond();
    //sumOnDevice_single_block <<<1, nElem>>>(d_A, d_B, d_C);
    //cudaDeviceReset();
    int threadInBlock = 1024;
    dim3 block(threadInBlock);
    dim3 grid ((nElem + block.x - 1) / block.x);
    sumOnDevice_multi_block <<<grid, block>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    double gpuTime = cpuSecond() - iStart;
    iStart = cpuSecond();
    sumOnHost(h_A, h_B, h_C, nElem);
    double cpuTime = cpuSecond() - iStart;
    CHECK(cudaMemcpy(res, d_C, nByte, cudaMemcpyDeviceToHost));
    
    printf("%f\t%f\n", h_C[0], h_C[nElem - 1]);
    printf("%f\t%f\n", res[0], res[nElem - 1]);
    printf("GPU: %lf\n", gpuTime);
    printf("CPU: %lf\n", cpuTime);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
    free(res);
}

#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        printf("Error: %s:%d, ", __FILE__, __LINE__);\
        printf("code:%d, reason: %s", error, cudaGetErrorString(error));\
        exit(-10 * error);\
    }\
}\


void initHostMatrix(float *h_A, int nxy)
{
    time_t t;
    srand((unsigned int) time(&t));
    for (int i = 0; i < nxy; ++i)
    {
        h_A[i] = (float) (rand() & 0xff) / 10.0f;
    }
}

void printMatrix(int *h_A, int nx, int ny)
{
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            printf("%d\t", h_A[i * nx + j]);
        }
        printf("\n");
    }
}

__global__ void printThreadIndex(int *d_A, const int nx, const int ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;
    
    printf("thread_id (%d, %d), block_id(%d, %d), coordinate(%d, %d), "
           "global index %d value %d\n", threadIdx.x, threadIdx.y,
           blockIdx.x, blockIdx.y, ix, iy, idx, d_A[idx]);
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            ic[j] = ia[j] + ib[j];
        }
        ia += nx;
        ib += nx;
        ic += nx;
    }
}

double getCpuSec()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

__global__ void sumMatrixOnDevice2D(float *A, float *B, float *C, const int x, const int y)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = blockIdx.y;
    unsigned int idx = iy * x + ix;

    C[idx] = A[idx] + B[idx];
}



int main(void)
{
    //get device info
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("using device %d : %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    //set matrix dimension
    int nx = 1024;
    int ny = 1024;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    //malloc host memory
    float *h_A, *h_B, *h_C, *res;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    h_C = (float*)malloc(nBytes);
    res = (float*)malloc(nBytes);

    //init host matrix
    initHostMatrix(h_A, nxy);
    initHostMatrix(h_B, nxy);
    //printMatrix(h_A, nx, ny);
    
    memset(h_C, 0, nBytes);
    memset(res, 0, nBytes);

    double start = getCpuSec();
    sumMatrixOnHost(h_A, h_B, h_C, nx, ny);
    printf("CPU: %lf\n", getCpuSec() - start);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    dim3 block(blockSize);
    dim3 grid((block.x + nx - 1) / block.x, ny);
    printf("block: (%d, %d), grid: (%d, %d)\n", block.x, block.y, grid.x, grid.y);

    start = getCpuSec();
    sumMatrixOnDevice2D<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
    printf("GPU: %lf\n", getCpuSec() - start);

    cudaDeviceSynchronize();

    //set up execution configuration
    
    cudaMemcpy(res, d_C, nBytes, cudaMemcpyDeviceToHost);
    printf("%f, %f\n%f, %f\n", h_C[0], res[0], h_C[nxy - 1], res[nxy - 1]);
    //invoke the kernel
    // printThreadIndex <<<grid, block>>>(d_A, nx, ny);

    //free host and device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(res);

    //reset device
    cudaDeviceReset();

    return 0;
}

#include <cuda_runtime.h>
#include <stdio.h>

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


void initHostMatrix(int *h_A, int nxy)
{
    for (int i = 0; i < nxy; ++i)
    {
        h_A[i] = i;
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


int main(void)
{
    //get device info
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("using device %d : %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    //set matrix dimension
    int nx = 8;
    int ny = 6;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(int);

    //malloc host memory
    int *h_A;
    h_A = (int*)malloc(nBytes);
    
    //init host matrix
    initHostMatrix(h_A, nxy);
    printMatrix(h_A, nx, ny);

    int *d_A;
    cudaMalloc((void **)&d_A, nBytes);

    //transfer data to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);

    //set up execution configuration
    dim3 block(4, 2);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    //invoke the kernel
    printThreadIndex <<<grid, block>>>(d_A, nx, ny);
    cudaDeviceSynchronize();

    //free host and device memory
    cudaFree(d_A);
    free(h_A);

    //reset device
    cudaDeviceReset();

    return 0;
}

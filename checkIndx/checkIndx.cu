#include <cuda_runtime.h>
#include <stdio.h>

__global__ void checkIndx(void)
{
    printf("threadIdx:(%d, %d, %d) "
           "blockIdx:(%d, %d, %d) "
           "blockDim:(%d, %d, %d) "
           "gridDim:(%d, %d, %d)\n",
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x, blockIdx.y, blockIdx.z,
           blockDim.x, blockDim.y, blockDim.z,
           gridDim.x, gridDim.y, gridDim.z);
}

int main()
{
    int nElem = 64;

    /**
      *here, we define 2 blocks, each of which has 3 threads.
      */
    for (int i = 1; i < 5; ++i)
    {
        if (i == 1 || i % 2 == 0) {
            dim3 block (nElem / i);
            dim3 grid  ((nElem + block.x - 1) / block.x);

            printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
            printf("block.x %d, block.y %d, block.z %d\n", block.x, block.y, block.z);
    
            checkIndx <<<grid, block>>>();
    
        }
    }
    cudaDeviceReset();

    return 0;
}

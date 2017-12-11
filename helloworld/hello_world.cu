#include <stdio.h>

__global__ void helloFromGPU(void)
{
   printf("hello world from GPU-%d!\n", threadIdx.x);

}

int main(void)
{
    printf("hello world from cpu!\n");
    helloFromGPU <<<1, 10>>>();
//    cudaDeviceReset();
    cudaDeviceSynchronize();
    return 0;

}

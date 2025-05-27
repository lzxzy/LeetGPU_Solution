#include "solve.h"
#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int numPixels = width * height;

    if (idx < numPixels){
        int pixelStart = idx * 4;
        image[pixelStart + 0] = 255 - image[pixelStart + 0];
        image[pixelStart + 1] = 255 - image[pixelStart + 1];
        image[pixelStart + 2] = 255 - image[pixelStart + 2];
    }
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
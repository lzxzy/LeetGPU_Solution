#include "solve.h"
#include <cuda_runtime.h>

__global__ void computeHistogramKernel(const int* input, int* histogram, int N, int num_bins){
    
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int gid = bid * blockDim.x + tid;

    if (gid < num_bins){
        int t_num = 0;
        for (int i=0; i< N; i ++){
            if(input[i] == gid){
                t_num += 1;
            }
        }
        histogram[gid] = t_num;
    }
}

// input, histogram are device pointers
void solve(const int* input, int* histogram, int N, int num_bins) {
    const int blockSize = 256;
    int numBlocks = (N + blockSize -1)/blockSize;
    computeHistogramKernel<<<numBlocks, blockSize>>>(input, histogram, N, num_bins);
    cudaDeviceSynchronize();
}

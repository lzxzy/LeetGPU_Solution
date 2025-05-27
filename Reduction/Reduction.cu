#include "solve.h"
#include <cuda_runtime.h>

__global__ void reduce_sum_kernel(const float* input, float * block_sums, int N){
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float x = (idx < N) ? input[idx] : 0.0f;
    sdata[tid] = x;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0 ; s>>=1){
        if(tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    //Write per-block result to global memory
    if(tid==0){
        block_sums[blockIdx.x] = sdata[0];
    }
}

void gpu_recursive_reduce(const float* d_input, float* d_output, int N){
    const int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks == 1){
        size_t sharedMemSize = threadsPerBlock * sizeof(float);
        reduce_sum_kernel<<<1, threadsPerBlock, sharedMemSize>>>(d_input, d_output, N);
        cudaDeviceSynchronize();
        return;
    }

    float* d_intermediate;
    cudaMalloc(&d_intermediate, blocks * sizeof(float));

    size_t sharedMemSize = threadsPerBlock * sizeof(float);
    reduce_sum_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(d_input, d_intermediate, N);
    cudaDeviceSynchronize();

    gpu_recursive_reduce(d_intermediate, d_output, blocks);
    cudaFree(d_intermediate);
}
// input, output are device pointers
// void solve(const float* input, float* output, int N) {  
//     const int threadsPerBlock = 256;
//     int blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;

//     float* d_block_sums;
//     cudaMalloc(&d_block_sums, blocksPerGrid * sizeof(float));

//     size_t sharedMemSize = threadsPerBlock * sizeof(float);
//     reduce_sum_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(input, d_block_sums, N);
//     cudaDeviceSynchronize();

//     //Second reduction if needed
//     if(blocksPerGrid > 1){
//         float * h_block_sums = new float[blocksPerGrid];
//         cudaMemcpy(h_block_sums, d_block_sums, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

//         float final_sum = 0.0f;
//         for(int i =0 ;i < blocksPerGrid; ++i){
//             final_sum += h_blocks_sums[i];
//         }
//         *output = final_sum;
//         delete[] h_block_sums;
//     }else{
//         cudaMemcpy(output, d_block_sums, sizeof(float), cudaMemcpyDeviceToDevice);
//     }
//     cudaFree(d_block_sums);
// }

void solve(const float* input, float* output, int N) {
    // input and output are device pointers
    gpu_recursive_reduce(input, output, N);
}

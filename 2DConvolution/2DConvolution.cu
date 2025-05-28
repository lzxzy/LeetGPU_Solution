#include "solve.h"
#include <cuda_runtime.h>

// input, kernel, output are device pointers
__global__ void conv2d_valid_kernel(const float* input, const float* kernel, float* output, 
                                    int input_rows, int input_cols,
                                    int kernel_rows, int kernel_cols,
                                    int output_rows, int output_cols){
    int out_r = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.x * blockDim.x + threadIdx.x;

    if(out_r < output_rows && out_c < output_cols){
        float sum = 0.0f;
        for(int kr = 0; kr < kernel_rows; ++kr){
            for(int kc = 0; kc < kernel_cols; ++kc){
                int in_r = out_r + kr;
                int in_c = out_c + kc;
                sum += input[in_r * input_cols + in_c] * kernel[kr * kernel_cols + kc];
            }
        }
        output[out_r * output_cols + out_c] = sum;
    }
}

void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    dim3 blockDim(16, 16);
    dim3 gridDim((output_cols + blockDim.x - 1)/blockDim.x,
                 (output_rows + blockDim.y - 1)/blockDim.y);

    conv2d_valid_kernel<<<gridDim, blockDim>>>(
        input, kernel, output,
        input_rows, input_cols,
        kernel_rows, kernel_cols,
        output_rows, output_cols
    );

    cudaDeviceSynchronize(); 
}
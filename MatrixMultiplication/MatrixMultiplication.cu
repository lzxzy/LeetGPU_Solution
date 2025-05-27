#include "solve.h"
#include <cuda_runtime.h>
#define BLOCK_SIZE 16

__device__ float GetElement(const float* A, int row, int col, int stride)
{
    return A[row * stride + col];
}

__device__ void SetElement(float* A, int row, int col, int stride, float value)
{
    A[row * stride + col] = value;
}

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Shared memory tiles
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Global row and col this thread computes
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float Cvalue = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // Load one tile of A and B into shared memory
        if (row < M && t * BLOCK_SIZE + threadIdx.x < N)
            As[threadIdx.y][threadIdx.x] = GetElement(A, row, t * BLOCK_SIZE + threadIdx.x, N);
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < K && t * BLOCK_SIZE + threadIdx.y < N)
            Bs[threadIdx.y][threadIdx.x] = GetElement(B, t * BLOCK_SIZE + threadIdx.y, col, K);
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Multiply the tiles
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[threadIdx.y][e] * Bs[e][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < K)
        SetElement(C, row, col, K, Cvalue);
}

void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((K + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
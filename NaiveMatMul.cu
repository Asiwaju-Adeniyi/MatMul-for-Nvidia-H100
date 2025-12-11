#include <cassert>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <utility>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

__global__ void naive_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int BLOCKSIZE = 32;

    const int row = blockIdx.x * blockDim.x + (threadIdx.x / BLOCKSIZE);
    const int col = blockIdx.y * blockDim.y + (threadIdx.x & BLOCKSIZE);

    if (row < M && col < N) {
        float acc = 0;

        for (int i = 0; i < K; ++i) {
            acc += A[row * K + i] * B[i * N + col];
        }

        C[row * N + col] = alpha * acc  + beta * C[row * N + col];
    }
}

int main() {
    int M = 4096;
    int N = 4096;
    int K = 4096;
    
    float alpha = 1.0f;
    float beta = 0.5f;



    float sizeA = M * K * sizeof(float);
    float sizeB = K * N * sizeof(float);
    float sizeC = M * N * sizeof(float);

    float *h_A = new float[M*K];
    float *h_B = new float[K*N];
    float *h_C = new float[M*N];

    float *d_A, *d_B;
    float *d_C;

    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
    dim3 blockDim(32,32, 1);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    naive_kernel<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);

    cudaMemcpy(h_C, d_A, sizeA, cudaMemcpyDeviceToHost);

    std::cout << "Done!" << std::endl;

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);






}

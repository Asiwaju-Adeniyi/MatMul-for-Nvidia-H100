#include <iostream>
#include <cuda_runtime.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

__global__ void naive_kernel(int M, int N, int K, float alpha,
                             const float *A, const float *B,
                             float beta, float *C) {
    int BLOCKSIZE = 32;

    int row = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    int col = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int i = 0; i < K; i++)
            acc += A[row * K + i] * B[i * N + col];

        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

int main() {
    int M = 4096;
    int N = 4096;
    int K = 4096;

    float alpha = 1.0f;
    float beta = 0.0f;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // Allocate host memory
    float *h_A = new float[M*K];
    float *h_B = new float[K*N];
    float *h_C = new float[M*N];

    // Initialize host memory
    for (int i = 0; i < M*K; i++) h_A[i] = 1;
    for (int i = 0; i < K*N; i++) h_B[i] = 1;
    for (int i = 0; i < M*N; i++) h_C[i] = 0;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sizeC, cudaMemcpyHostToDevice);

    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(1024);

    naive_kernel<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    std::cout << "Done!" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}

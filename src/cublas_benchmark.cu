#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <assert.h>

#include <vector>
#include <iostream>
#include <numeric>
#include <chrono>
#include <thread>

inline half RAND_HALF(float LO = -1.0f, float HI = 1.0f)
{
    float r = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
    return (half) r;
}

int main(int argc, char *argv[]){
    const unsigned int M = 8192;
    const unsigned int N = 8192;
    const unsigned int K = 8192;
    const half alpha = __float2half(0.7f);
    const half beta = __float2half(0.3f);
    int num_runs = 50;

    half *A, *B, *C;
    A = (half *)malloc(M * K * sizeof(half));
    B = (half *)malloc(K * N * sizeof(half));
    C = (half *)malloc(M * N * sizeof(half));

    half *dev_A, *dev_B, *dev_C;
    cudaMalloc((void **)&dev_A, M * K * sizeof(half));
    cudaMalloc((void **)&dev_B, K * N * sizeof(half));
    cudaMalloc((void **)&dev_C, M * N * sizeof(half));

    for (int i = 0; i < M * N; i++) {C[i] = RAND_HALF();}
    for (int i = 0; i < K * N; i++) {B[i] = RAND_HALF();}
    for (int i = 0; i < M * K; i++) {A[i] = RAND_HALF();}

    cudaMemcpy(dev_A, A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, K * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_C, C, M * N * sizeof(half), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    auto run_gemm = [&]() {
        cublasGemmEx(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            dev_A, CUDA_R_16F, K,  // A: FP16, leading dim K
            dev_B, CUDA_R_16F, K,  // B: FP16, leading dim K
            &beta,
            dev_C, CUDA_R_16F, M,  // C: FP16, leading dim M
            CUBLAS_COMPUTE_32F,    // compute type: FP32 accumulation
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
    };

    //warmup
    run_gemm();
    cudaDeviceSynchronize();

    //main loop
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    std::vector<float> times;
    float elapsed;

    for (int i=0; i<num_runs; ++i){
        cudaEventRecord(start);
        run_gemm();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        times.push_back(elapsed);
    }

    double avg_time_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double total_flops = 2.0 * M * N * K;
    double gflops_per_sec = (total_flops) / (avg_time_ms * 1.0e6);
    times.clear();
    std::cout << "cuBLAS: " << gflops_per_sec << " GFLOPS/sec for " << M << "x" << N << "x" << K << std::endl;
    cublasDestroy(handle);
    cudaFree(dev_A); cudaFree(dev_B); cudaFree(dev_C);
    free(A); free(B); free(C);

    return 0;
}
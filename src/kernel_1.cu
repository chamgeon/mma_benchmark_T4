//no mma atom, using universal fma, equivalent to:
//   CUTE_UNROLL
//   for (int k = 0; k < size<1>(tCsA); ++k) {
//     CUTE_UNROLL
//     for (int m = 0; m < size<0>(tCrC); ++m) {
//       CUTE_UNROLL
//       for (int n = 0; n < size<1>(tCrC); ++n) {
//         tCrC(m,n) += tCsA(m,k) * tCsB(n,k);
//       }
//     }
//   }

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <vector>
#include <iostream>
#include <numeric>
#include <chrono>
#include <thread>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>


template <class TABC, class glayoutA, class glayoutB, class glayoutC,
          class CTAtiler, class slayoutA, class slayoutB,
          class tlayoutA, class tlayoutB, class tlayoutC>
__launch_bounds__(decltype(size(tlayoutC{}))::value)
__global__ void mma_naive_kernel(
    TABC* A, TABC* B, TABC* C, TABC alpha, TABC beta,
    glayoutA gl_A, glayoutB gl_B, glayoutC gl_C, CTAtiler cta_tiler,
    slayoutA sl_A, slayoutB sl_B,
    tlayoutA tl_A, tlayoutB tl_B, tlayoutC tl_C
){
    using namespace cute;
    Tensor tensor_A = make_tensor(make_gmem_ptr(A), gl_A);   //(M,K)
    Tensor tensor_B = make_tensor(make_gmem_ptr(B), gl_B);   //(N,K)
    Tensor tensor_C = make_tensor(make_gmem_ptr(C), gl_C);   //(M,N)

    auto const cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(tensor_A, cta_tiler, cta_coord, Step<_1, X,_1>{});   //(blk_M, blk_K, k)
    Tensor gB = local_tile(tensor_B, cta_tiler, cta_coord, Step<X, _1,_1>{});   //(blk_N, blk_K, k)
    Tensor gC = local_tile(tensor_C, cta_tiler, cta_coord, Step<_1, _1,X>{});   //(blk_M, blk_N)
    Tensor tAgA = local_partition(gA, tl_A, threadIdx.x);   //(thr_M_cp, thr_K_cp, k)
    Tensor tBgB = local_partition(gB, tl_B, threadIdx.x);   //(thr_N_cp, thr_K_cp, k)

    __shared__ TABC smemA[cosize_v<slayoutA>];
    __shared__ TABC smemB[cosize_v<slayoutB>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sl_A);   //(blk_M, blk_K)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sl_B);   //(blk_N, blk_K)
    Tensor tAsA = local_partition(sA, tl_A, threadIdx.x);   //(thr_M_cp, thr_K_cp)
    Tensor tBsB = local_partition(sB, tl_B, threadIdx.x);   //(thr_N_cp, thr_K_cp)

    Tensor tCsA = local_partition(sA, tl_C, threadIdx.x, Step<_1, X>{});   //(thr_M_mma, blk_K)
    Tensor tCsB = local_partition(sB, tl_C, threadIdx.x, Step<X, _1>{});   //(thr_N_mma, blk_K)
    Tensor tCgC = local_partition(gC, tl_C, threadIdx.x);   //(thr_M_mma, thr_N_mma)
    Tensor tCrC = make_tensor<float>(shape(tCgC), LayoutLeft{});   //fp32 accumulate

    clear(tCrC);

    for (int k_tile=0; k_tile < size<2>(tAgA); ++k_tile){
        copy(tAgA(_,_,k_tile), tAsA);
        copy(tBgB(_,_,k_tile), tBsB);
        __syncthreads();

        gemm(tCsA, tCsB, tCrC);
        __syncthreads();
    }

    axpby(alpha, tCrC, beta, tCgC);
}



template <int M, int N, int K, class TABC>
void gemm_cpu_reference(
    TABC* A, TABC* B, TABC* C,
    TABC alpha, TABC beta)
{
    using namespace cute;
    auto const shape_M = Int<M>{};
    auto const shape_N = Int<N>{};
    auto const shape_K = Int<K>{};

    auto const gmem_shape_A = make_shape(shape_M,shape_K);
    auto const gmem_stride_A = make_stride(shape_K, Int<1>{});    //K-major
    auto const gmem_layout_A = make_layout(gmem_shape_A, gmem_stride_A);

    auto const gmem_shape_B = make_shape(shape_N,shape_K);
    auto const gmem_stride_B = make_stride(shape_K, Int<1>{});    //K-major
    auto const gmem_layout_B = make_layout(gmem_shape_B, gmem_stride_B);

    auto const gmem_shape_C = make_shape(shape_M, shape_N);
    auto const gmem_stride_C = make_stride(Int<1>{}, shape_M);    //M-major
    auto const gmem_layout_C = make_layout(gmem_shape_C, gmem_stride_C);

    Tensor tensor_A = make_tensor(make_gmem_ptr(A), gmem_layout_A);   //(M,K)
    Tensor tensor_B = make_tensor(make_gmem_ptr(B), gmem_layout_B);   //(N,K)
    Tensor tensor_C = make_tensor(make_gmem_ptr(C), gmem_layout_C);   //(M,N)

    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += static_cast<float>(tensor_A(m,k) * tensor_B(n,k));
            tensor_C(m,n) = static_cast<TABC>(
                float(alpha) * acc + float(beta) * float(tensor_C(m,n))
            );   //i'm a lulu baby
        }

}



int main(int argc, char** argv){

    using namespace cute;
    using TABC = half_t;

    constexpr int M{8192};
    constexpr int N{8192};
    constexpr int K{8192};
    
    //for evaluation
    //constexpr int M{512};
    //constexpr int N{512};
    //constexpr int K{256};

    auto const gmem_size_A = M*K;
    auto const gmem_size_B = N*K;
    auto const gmem_size_C = M*N;
    auto h_gmem_A = thrust::host_vector<TABC>(gmem_size_A);
    auto h_gmem_B = thrust::host_vector<TABC>(gmem_size_B);
    auto h_gmem_C = thrust::host_vector<TABC>(gmem_size_C);
    auto d_gmem_A = thrust::device_vector<TABC>(gmem_size_A);
    auto d_gmem_B = thrust::device_vector<TABC>(gmem_size_B);
    auto d_gmem_C = thrust::device_vector<TABC>(gmem_size_C);

    for (int i=0; i<gmem_size_A; ++i) h_gmem_A[i] = static_cast<TABC>(2*(rand()/double(RAND_MAX))-1);
    for (int i=0; i<gmem_size_B; ++i) h_gmem_B[i] = static_cast<TABC>(2*(rand()/double(RAND_MAX))-1);
    for (int i=0; i<gmem_size_C; ++i) h_gmem_C[i] = static_cast<TABC>(2*(rand()/double(RAND_MAX))-1);
    d_gmem_A = h_gmem_A;
    d_gmem_B = h_gmem_B;
    d_gmem_C = h_gmem_C;

    TABC alpha = static_cast<TABC>(0.7);
    TABC beta = static_cast<TABC>(0.3);

    //layouts
    auto const shape_M = Int<M>{};
    auto const shape_N = Int<N>{};
    auto const shape_K = Int<K>{};

    auto const gmem_shape_A = make_shape(shape_M,shape_K);
    auto const gmem_stride_A = make_stride(shape_K, Int<1>{});    //K-major
    auto const gmem_layout_A = make_layout(gmem_shape_A, gmem_stride_A);

    auto const gmem_shape_B = make_shape(shape_N,shape_K);
    auto const gmem_stride_B = make_stride(shape_K, Int<1>{});    //K-major
    auto const gmem_layout_B = make_layout(gmem_shape_B, gmem_stride_B);

    auto const gmem_shape_C = make_shape(shape_M, shape_N);
    auto const gmem_stride_C = make_stride(Int<1>{}, shape_M);    //M-major
    auto const gmem_layout_C = make_layout(gmem_shape_C, gmem_stride_C);

    auto const shape_blk_M = Int<128>{};
    auto const shape_blk_N = Int<128>{};
    auto const shape_blk_K = Int<8>{};
    auto const cta_tiler = make_shape(shape_blk_M, shape_blk_N, shape_blk_K);

    auto const smem_shape_A = make_shape(shape_blk_M, shape_blk_K);
    auto const smem_shape_B = make_shape(shape_blk_N, shape_blk_K);
    auto const smem_stride_A = make_stride(shape_blk_K, Int<1>{});  //K-major
    auto const smem_stride_B = make_stride(shape_blk_K, Int<1>{});  //K-major
    auto const smem_layout_A = make_layout(smem_shape_A, smem_stride_A);
    auto const smem_layout_B = make_layout(smem_shape_B, smem_stride_B);

    auto const thread_shape_A = make_shape(Int<32>{}, Int<8>{});
    auto const thread_shape_B = make_shape(Int<32>{}, Int<8>{});
    auto const thread_shape_C = make_shape(Int<16>{}, Int<16>{});
    auto const thread_stride_A = make_stride(Int<8>{}, Int<1>{});  //K-major
    auto const thread_stride_B = make_stride(Int<8>{}, Int<1>{});  //K-major
    auto const thread_stride_C = make_stride(Int<1>{}, Int<16>{}); //M-major
    auto const thread_layout_A = make_layout(thread_shape_A, thread_stride_A);
    auto const thread_layout_B = make_layout(thread_shape_B, thread_stride_B);
    auto const thread_layout_C = make_layout(thread_shape_C, thread_stride_C);

    dim3 gridDim(size(ceil_div(shape_M, shape_blk_M)), size(ceil_div(shape_N, shape_blk_N)));
    dim3 blockDim(size(thread_layout_A));

    auto run_gemm = [&]() {
        mma_naive_kernel<<< gridDim, blockDim >>>(
            d_gmem_A.data().get(), d_gmem_B.data().get(), d_gmem_C.data().get(), 
            alpha, beta,
            gmem_layout_A, gmem_layout_B, gmem_layout_C,
            cta_tiler,
            smem_layout_A, smem_layout_B,
            thread_layout_A, thread_layout_B, thread_layout_C
        );
    };

    //evaluate
    #if 0
    auto h_gmem_C_ref = h_gmem_C;

    run_gemm();
    h_gmem_C = d_gmem_C;
    gemm_cpu_reference<M,N,K>(
        thrust::raw_pointer_cast(h_gmem_A.data()),
        thrust::raw_pointer_cast(h_gmem_B.data()),
        thrust::raw_pointer_cast(h_gmem_C_ref.data()),
        alpha, beta);

    float max_error = 0.0f;
    for (int i = 0; i < gmem_size_C; ++i) {
        float diff = abs(float(h_gmem_C[i]) - float(h_gmem_C_ref[i]));
        max_error = max(max_error, diff);
    }
    printf("Max error: %f\n", max_error);
    if (max_error < 2e-2f){
        printf("evaluation PASSED\n");
    }
    else{
        printf("evaluation FAILED\n");
    }
    #endif


    //benchmark
    #if 1
    //warmup
    run_gemm();
    cudaDeviceSynchronize();

    //main loop
    int num_runs = 50;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    std::vector<float> times;

    for (int i=0; i<num_runs; ++i){
        cudaEventRecord(start,0);
        run_gemm();
        cudaEventRecord(stop,0);
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        times.push_back(elapsed);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    double avg_time_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double total_flops = 2.0 * M * N * K;
    double gflops_per_sec = (total_flops) / (avg_time_ms * 1.0e6);
    times.clear();
    std::cout << "kernel_1: " << gflops_per_sec << " GFLOPS/sec for " << M << "x" << N << "x" << K << std::endl;
    #endif

    return 0;
}
//more arithmetic intensity

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
          class TiledCopyGS, class TiledCopySRA, class TiledCopySRB, class TiledMMA>
__global__ static
__launch_bounds__(decltype(size(TiledCopyGS{}))::value)
void tiled_mma_kernel(
    const TABC* __restrict__ A, const TABC* __restrict__ B, TABC* __restrict__ C,
    TABC alpha, TABC beta,
    glayoutA gl_A, glayoutB gl_B, glayoutC gl_C,
    CTAtiler cta_tiler, slayoutA sl_A, slayoutB sl_B,
    TiledCopyGS copy_gs, TiledCopySRA copy_sr_A, TiledCopySRB copy_sr_B, TiledMMA mma
) {
    using namespace cute;
    Tensor tensor_A = make_tensor(make_gmem_ptr(A), gl_A);
    Tensor tensor_B = make_tensor(make_gmem_ptr(B), gl_B);
    Tensor tensor_C = make_tensor(make_gmem_ptr(C), gl_C);

    auto const cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(tensor_A, cta_tiler, cta_coord, Step<_1, X,_1>{});   // (bM, bK, k)
    Tensor gB = local_tile(tensor_B, cta_tiler, cta_coord, Step<X, _1,_1>{});   // (bN, bK, k)
    Tensor gC = local_tile(tensor_C, cta_tiler, cta_coord, Step<_1, _1,X>{});   // (bM, bN)

    __shared__ TABC smemA[cosize_v<slayoutA>];
    __shared__ TABC smemB[cosize_v<slayoutB>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sl_A);   // (bM, bK)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sl_B);   // (bN, bK)

    ThrCopy thr_copy_gs = copy_gs.get_thread_slice(threadIdx.x);
    Tensor thr_gs_gA = thr_copy_gs.partition_S(gA);   // (copy gs atom val, block-tile layout, k)
    Tensor thr_gs_gB = thr_copy_gs.partition_S(gB);   // (copy gs atom val, block-tile layout, k)
    Tensor thr_gs_sA = thr_copy_gs.partition_D(sA);   // (copy gs atom val, block-tile layout)
    Tensor thr_gs_sB = thr_copy_gs.partition_D(sB);   // (copy gs atom val, block-tile layout)
    Tensor thr_gs_rA = make_fragment_like(thr_gs_sA);
    Tensor thr_gs_rB = make_fragment_like(thr_gs_sB);

    //prefetch
    copy(copy_gs, thr_gs_gA(_,_,_,0), thr_gs_rA);
    copy(copy_gs, thr_gs_gB(_,_,_,0), thr_gs_rB);

    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
    Tensor thr_mma_rA = thr_mma.partition_fragment_A(sA);   // (mma atom val A, block-cta layout A)
    Tensor thr_mma_rB = thr_mma.partition_fragment_B(sB);   // (mma atom val B, block-cta layout B)
    Tensor thr_mma_gC = thr_mma.partition_C(gC);
    Tensor thr_mma_rC = thr_mma.make_fragment_C(thr_mma_gC);   // (mma atom val C, block-cta layout C)

    ThrCopy thr_copy_sr_A = copy_sr_A.get_thread_slice(threadIdx.x);
    ThrCopy thr_copy_sr_B = copy_sr_B.get_thread_slice(threadIdx.x);
    Tensor thr_sr_sA = thr_copy_sr_A.partition_S(sA);   // (copy sr atom val, block-cta layout)
    Tensor thr_sr_sB = thr_copy_sr_B.partition_S(sB);   // (copy sr atom val, block-cta layout)
    Tensor thr_sr_rA = thr_copy_sr_A.retile_D(thr_mma_rA);   // (copy sr atom val, block-cta layout)
    Tensor thr_sr_rB = thr_copy_sr_B.retile_D(thr_mma_rB);   // (copy sr atom val, block-cta layout)

    clear(thr_mma_rC);

    copy(thr_gs_rA, thr_gs_sA);
    copy(thr_gs_rB, thr_gs_sB);
    __syncthreads();
    
    //main loop
    auto k_tile_num = size<3>(thr_gs_gA);

    CUTE_UNROLL
    for(int i=1; i<k_tile_num; ++i){
        copy(copy_gs, thr_gs_gA(_,_,_,i), thr_gs_rA);
        copy(copy_gs, thr_gs_gB(_,_,_,i), thr_gs_rB);

        copy(copy_sr_A, thr_sr_sA, thr_sr_rA);
        copy(copy_sr_B, thr_sr_sB, thr_sr_rB);
        __syncthreads();

        gemm(mma, thr_mma_rA, thr_mma_rB, thr_mma_rC);

        copy(thr_gs_rA, thr_gs_sA);
        copy(thr_gs_rB, thr_gs_sB);
        __syncthreads();
    }

    //epilogue - last ktile and axpby
    copy(copy_sr_A, thr_sr_sA, thr_sr_rA);
    copy(copy_sr_B, thr_sr_sB, thr_sr_rB);
    __syncthreads();

    gemm(mma, thr_mma_rA, thr_mma_rB, thr_mma_rC);
    __syncthreads();

    axpby(alpha, thr_mma_rC, beta, thr_mma_gC);
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
            );
        }
}




int main(int argc, char** argv){
    using namespace cute;
    using TABC = half_t;
    using CopyOP_GS = UniversalCopy<uint128_t>;
    using CopyOP_SR = SM75_U32x2_LDSM_N;
    using MMAOP = SM75_16x8x8_F32F16F16F32_TN;

    constexpr int M{8192};
    constexpr int N(8192);
    constexpr int K{8192};
    //for correctness test
    //constexpr int M{512};
    //constexpr int N{512};
    //constexpr int K{256};
    constexpr int bM{128};
    constexpr int bN{256};
    constexpr int bK{64};
    
    constexpr int gmem_size_A = M*K;
    constexpr int gmem_size_B = N*K;
    constexpr int gmem_size_C = M*N;

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
    auto const shape_bM = Int<bM>{};
    auto const shape_bN = Int<bN>{};
    auto const shape_bK = Int<bK>{};

    auto const gmem_shape_A = make_shape(shape_M, shape_K);
    auto const gmem_shape_B = make_shape(shape_N, shape_K);
    auto const gmem_shape_C = make_shape(shape_M, shape_N);
    auto const gmem_stride_A = make_stride(shape_K, Int<1>{});   //K major
    auto const gmem_stride_B = make_stride(shape_K, Int<1>{});   //K major
    auto const gmem_stride_C = make_stride(Int<1>{}, shape_M);   //M major
    auto const gmem_layout_A = make_layout(gmem_shape_A,gmem_stride_A);
    auto const gmem_layout_B = make_layout(gmem_shape_B,gmem_stride_B);
    auto const gmem_layout_C = make_layout(gmem_shape_C,gmem_stride_C);
    
    auto const smem_shape_A = make_shape(shape_bM, shape_bK);
    auto const smem_shape_B = make_shape(shape_bN, shape_bK);
    auto const smem_stride_A = make_stride(shape_bK, Int<1>{});   //K major
    auto const smem_stride_B = make_stride(shape_bK, Int<1>{});   //K major
    auto const swizzle = Swizzle<3, 3, 3>{};
    auto const smem_layout_A = composition(
        swizzle,
        make_layout(smem_shape_A, smem_stride_A)
    );
    auto const smem_layout_B = composition(
        swizzle,
        make_layout(smem_shape_B, smem_stride_B)
    );
    auto const cta_tiler = make_shape(shape_bM, shape_bN, shape_bK);

    auto const copy_gs_thread_shape = make_shape(Int<32>{}, Int<8>{});
    auto const copy_gs_thread_stride = make_stride(Int<8>{}, Int<1>{});
    auto const copy_gs_thread_layout = make_layout(copy_gs_thread_shape, copy_gs_thread_stride);
    auto const copy_gs_val_shape = make_shape(Int<1>{}, Int<8>{});
    auto const copy_gs_val_layout = make_layout(copy_gs_val_shape);

    auto const mma_warps_shape = make_shape(Int<4>{}, Int<2>{}, Int<1>{});   ///4x2x1 atoms per cta
    auto const mma_warps_layout = make_layout(mma_warps_shape);
    auto const mma_tile = make_tile(Int<64>{}, Int<32>{}, Int<8>{});

    //atom, tiledcopy, tiledmma, dims

    Copy_Atom<CopyOP_GS, TABC> copy_atom_gs;
    Copy_Atom<CopyOP_SR, TABC> copy_atom_sr;
    MMA_Atom<MMAOP> mma_atom;

    auto const copy_gs = make_tiled_copy(copy_atom_gs, copy_gs_thread_layout, copy_gs_val_layout);
    auto const mma = make_tiled_mma(mma_atom, mma_warps_layout, mma_tile);
    auto const copy_sr_A = make_tiled_copy_A(copy_atom_sr, mma);
    auto const copy_sr_B = make_tiled_copy_B(copy_atom_sr, mma);

    dim3 gridDim(size(ceil_div(shape_M, shape_bM)), size(ceil_div(shape_N, shape_bN)));
    dim3 blockDim(size(copy_gs_thread_layout));

    auto run_gemm = [&]() {
        tiled_mma_kernel<<<gridDim, blockDim>>>(
            d_gmem_A.data().get(), d_gmem_B.data().get(), d_gmem_C.data().get(), 
            alpha, beta,
            gmem_layout_A, gmem_layout_B, gmem_layout_C,
            cta_tiler, smem_layout_A, smem_layout_B,
            copy_gs, copy_sr_A, copy_sr_B, mma
        );
    };


    //correctness
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
        printf("evaluation PASSED, now benchmarking\n");
    } else {
        printf("evaluation FAILED\n");
    } // 1 ULP of fp16 ~= 0.015625 at values near 16

    #endif


    //warmup
    run_gemm();
    cudaDeviceSynchronize();

    #if 1
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
    std::cout << "kernel_5: " << gflops_per_sec << " GFLOPS/sec for " << M << "x" << N << "x" << K << std::endl;
    #endif

    return 0;
}

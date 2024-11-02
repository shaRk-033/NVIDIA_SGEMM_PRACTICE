![](images/head.png)

![](https://img.shields.io/badge/build-passing-brightgreen) ![](https://img.shields.io/badge/ubuntu-18.04-blue) ![](https://img.shields.io/badge/cuda-10.2-blue) ![](https://img.shields.io/badge/nvidia-RTX3090-blue) ![](https://img.shields.io/badge/cmake-3.21-blue)



# Overview

For NVIDIA GPUs, use CUDA programming to gradually optimize the performance of matrix multiplication operations：

| Kernel function | Description | GFLOPS | Custom kernel function / CUBLAS(%) |
| -------- | ----------------------- | -------- | ------------------------ |
| CUBLAS | Official library functions | 14448.69 / Benchmark |
| kernel_1 / Simple implementation | 2262.168 | 15.65657 |
| kernel_2 | Shared memory cache | 4216.536 | 29.18283 |
| kernel_3 | One-dimensional Thread Tile Parallel optimization | 7809.629 | 54.05078 |
| kernel_4 | Two-dimensional Thread Tile Parallel optimization | 12251.3 | 84.79179 |
| kernel_5 / Register cache | 12177.95 | 84.28412 |
| kernel_6 | FLOAT4 vector memory access | 13161.49 | 91.09125 |
| kernel_7 | Dual cache prefetch | 13634.98 | 94.36832 |

> NVIDIA GeForce RTX 3090, matrix size 5120

# Configure

-Compiled using 'gcc 7.5.0` under Ubuntu 18.04.5 LTS
- NVIDIA CUDA version: `CUDA 10.2`；

# Table of Contents

```
NVIDIA_SGEMM_PRACTICE # Root directory
    ─── images # picture result
    │     ├── describe_kernel_1.png  
    │     ├── describe_kernel_x.png
    │     └── kernel_x_vs_y.png
    ─── test # test result
    │     ├── test_kernel_0.txt 
    │     ├── test_kernel_1.txt 
    │     └── test_kernel_x.txt 
    ───src # source file
    │    ├── kernel
    │    │  ├── kernel_1.cuh # Declaration and definition
    │    │  ├── kernel_2.cuh
    │    │  └── kernel_x.cuh
    │    ├── kernel.cuh
    │    ├── utils.cuh # helper function
    │    └── utils.cu
    ─── plot.py # Drawing based on test results
    ─── run.sh # Run the compiled executable file
    ─── sgemm.cu # Main program
    └── CMakeLists.txt # Compilation related
```

# Run
1. Configure NVCC compilation parameters
> In CMakeLists.Modify'set(CUDA_NVCC_FLAGS-arch=compute_70;-code=compute_70)` in txt
2. Configure the matrix to calculate the maximum size
> Modify'size_len` in'sgemm.cu:16`, it is recommended to set it to 16 for the initial operation. Excessive size may cause the power supply to be overloaded and the host to restart.；
3. compile
`cd build && cmake .. && make`
4. 运行run.sh , Count the calculation efficiency of each kernel function, and save the results in the test directory；
5. Calculation efficiency polyline drawing

>`python plot.py 0 1` means drawing a comparison chart of the computational efficiency of CUBLAS and kernel_1；

# Step-by-step optimization

##  kernel 1 

**Naive Basic Version Matrix multiplication Implementation**

Correspond each logical thread to each element of matrix C, and each thread is responsible for the calculation of an element in C；

![](./images/describe_kernel_1.png)

```cpp
__global__ __launch_bounds__(1024) void
mysgemm_v1(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {

    int gx = blockIdx.x * blockDim.x + threadIdx.x; // Global x
    int gy = blockIdx.y * blockDim.y + threadIdx.y; // Global y

    float tmp = 0.;
    for (int i = 0; i < K; i++) {
        tmp +=A[gy*K + i] * B[i* N + gx]; // Two global memory accesses and one FMA (accumulation and multiplication)
    }
    C[gy * N + gx] = alpha * tmp + beta * C[gy * N + gx];
}
```

![](./images/kernel_culas_vs_1.png)

The unoptimized matrix multiplication performance is less than 1/10 of CUBLAS. The specific analysis is as follows；

-Calculate the memory access ratio: each iteration requires one FMA (multiplication and accumulation) and two global memory reads to calculate the memory access ratio of 1/2；
-Access to stock: Access to global memory, each element of the C matrix needs to access `2K` single-precision floating-point numbers for calculation, and `2*K*M*N` is required to complete all calculations.；

Global memory access latency is high (several hundred cycles), and elements in the same location are read repeatedly at the same time (the same row element in C is calculated to share the same row element in A, and the same column element in C is calculated to share the same column element in B). On the other hand, the lower calculated memory access ratio cannot effectively hide the memory access delay. Therefore, the memory access delay and the calculated memory access ratio are the reasons for the inefficiency of kernel 1.

## kernel 2

**Use shared memory cache to reduce global memory access and memory access latency**

The memory access delay comes from the high latency of global memory and repeated access to global memory.Shared memory is on-chip memory, which has a low memory access delay (dozens of cycles). Using shared memory for caching can reduce memory access delay.；

![](./images/describe_kernel_2.png)

> BM and BN represent the height and width of the block tile, and BK represents the step size of the global memory to be cached, that is, the calculation of a block needs to be cached K/BK times；

Shared memory caches the global memory A tile and B tile, completes the FMA calculation of all elements in the C block, and continuously slides the cache area to update the block.；

```cpp
/*
dim3 blockDim(1024);
dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
mysgemm_v2<32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
*/

template<const int BLOCK_SIZE>
__global__ void mysgemm_v2(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    const int BM = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;
    
    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN;

    // Apply for shared memory space
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move to the current block
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    float tmp = 0.;
    for (int k = 0; k < K; k += BK) {
        // Cache A_tile and B_tile
        As[ty * BK + tx] = A[ty * K + tx];
        Bs[ty * BN + tx] = B[ty * N + tx];
        // Synchronize all thread caches to complete
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int i = 0; i < BK; i++) {
            tmp += As[ty * BK + i] * Bs[i * BN + tx];
        }
        // FMA calculation needs to read the cache data and synchronize it before a new round of writing to the cache to ensure that all thread calculations are completed
        __syncthreads();
    }
    C[ty * N + tx] = alpha * tmp + beta * C[ty * N + tx];
}
```

![](./images/kernel_1_vs_2.png)

-Access to stock: each block needs to read`(K/BK)*(BM*BK+BK*BN)` single-precision floating-point numbers from global memory, and there are`(M/BM)*(N/BN)` blocks in the entire C, so to complete the calculation of all elements in C, you need to read`(M/BM)*(N/BN)*(K/BK)*(BM*BK+BK*BN)' single-precision floating-point numbers

Kernel 1 is limited by the memory access delay and repeated access of global memory. Before optimization, the global memory access is `2*K*M*N`. After the shared memory cache is optimized, the memory access is reduced to the original `1/2*(1/BN)*(1/BM)`, when`BN=BM=32`, the memory access is reduced to 1/32; on the other hand, the shared memory memory access delay is much lower than that of global memory, so the computing efficiency has been improved to a certain extent.

## kernel 3

**Optimize with one-dimensional thread tile**

It is known that the number of visits to global memory can be further reduced by increasing the block size (BM, BN) value, so BM and BN can be increased from 32 to 64.；

>** Can global memory access be reduced by infinitely increasing the block size?**
>
> No, on the one hand, the size of the block block matrix is too large and the number of blocks is reduced, which will cause a large number of SM (Streaming Multiprocessor) idle waste; on the other hand, the increase of BN and BM requires more shared memory to be applied for. The more shared memory in a single thread, the less active thread bundles, which is not conducive to hiding instruction delays.；

Therefore, while increasing the BM and BN values, in order to reduce the shared memory footprint, on the one hand, the BK value is reduced to 8.；

> When increasing the block size, special attention should be paid to the consumption of shared memory, limit the size of shared memory and the number of threads in the block, and avoid the inability to start the kernel function due to insufficient resources.

![](./images/describe_kernel_3_1.png)

On the other hand, the shared memory cache reduces the global memory access to stock and the memory access delay accumulated by FMA multiplication, but the calculation of the memory access ratio has not been improved. Each iteration of the calculation requires two memory access instructions and one calculation instruction. Therefore, thread tile is introduced, that is, a thread is responsible for the calculation of multiple elements in the block, TM and TN represent the height and width of the thread tile respectively.

![](./images/describe_kernel_3_2.png)

```cpp
/*
dim3 blockDim(512);
dim3 gridDim(CEIL_DIV(M, 64), CEIL_DIV(N, 64));
mysgemm_v3<64, 64, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
*/


template<const int BM,
        const int BN,
        const int BK,
        const int TM>
__global__ void mysgemm_v3(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int thread_num= BM* BN/TM; // A thread is responsible for calculating TM elements in the block

    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN * TM;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move to the current block
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    /*
    The current thread is responsible for moving the a_tile_row row and a_tile_col column elements in the global memory to the a_tile_row row and a_tile_col column in the shared memory.
    a_tile_stride means that the thread in the block can carry the a_tile_stride line to shared memory；

    If BM=64, BK=8, thread_num=512, then a_tile_stride=64, a_tile_stride=BM, it means that each thread can complete the handling of the required elements in one round.;
    If BM=128, BK=8, thread_num=512, then a_tile_stride=64, which means that each thread can carry two rounds to complete the handling of the required elements.;
    */
    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

    float tmp[TM + 1] = {0.}; // Each thread is responsible for TM elements, you need to apply for TM registers to save the accumulated value, and an additional register is used for caching；
    #pragma unroll
    for (int k = 0; k < K; k += BK) {
        #pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
        }
        #pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
        }
        __syncthreads();
        A += BK;
        B += BK * N;
        #pragma unroll
        for (int i = 0; i < BK; i++) {
            tmp[TM] =Bs[tx+i*BN]; // An additional register to avoid repeatedly reading Bs[tx+i *BN] from shared memory
            #pragma unroll// Loop expansion, increase instruction parallelism
            for (int j = 0; j < TM; j++) {
                tmp[j] += As[(ty + j) * BK + i] * tmp[TM];
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < TM; j++) {
        C[(ty + j) * N + tx] = alpha * tmp[j] + beta * C[(ty + j) * N + tx];
    }
}
```

![](./images/kernel_2_vs_3.png)

This example is optimized from two aspects：

-Global memory access to stock: Compared with the initial version, by caching the `64*64' block size, the access to stock is reduced to 1/64；
-Calculate the memory access ratio: introduce thread tile, use a single thread to be responsible for the calculation of multiple elements, and increase the calculated memory access ratio; when TM=8, for each execution of 8 memory access instructions for shared memory As and 1 memory access instruction for shared memory Bs, 8 calculation instructions can be executed, compared with the initial version of the calculation, the memory access ratio is 1:2, increased to 8:9, effectively hiding the memory access delay；

Through the two-aspect optimization of this example, the calculation efficiency of matrix multiplication is significantly increased by nearly twice；

## kernel 4

**Optimized with two-dimensional thread tile**

Set the thread tile to two-dimensional, that is, a thread is responsible for the calculation of a small piece of element, thereby further increasing the block size and reducing the number of global memory accesses.；

> Increase the thread tile size, you can calculate a larger block size with the same number of threads or fewer threads;

More importantly, a single thread is responsible for calculating more C element areas, which can increase the degree of instruction-level parallelism.；

> Why can the degree of instruction parallelism be improved?
>
> The more instructions processed by a single thread, the longer the pipeline level. Since a single-threaded pipeline can process multiple instructions in parallel, although the execution of a single instruction slows down, the number of instructions processed per unit time becomes larger, which improves throughput and hides instruction delays; instruction-level concurrency has more advantages than thread-level concurrency.

![](./images/describe_kernel_4.png)

Set up a thread to be responsible for the calculation of elements in the 8×8 area, that is, thread tile=8×8, TM=8, TN=8；

```cpp
//BM=BN=128, BK=8, TM=TN=8, shared memory size 128*8
dim3 blockDim(256);
dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 128));
mysgemm_v4<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);

    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride= thread_num/BK; // 128*8/256 =4, all threads need to be transported for 4 rounds, and the 128*8 size area in the global memory can be transported to the shared memory

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

// Each thread is responsible for TM*TN elements, you need to apply for TM*TN registers to save the accumulated value；
float tmp[TM][TN] = {0.}; 

// A single thread loop TM, TN completes the multiplication and accumulation of elements in the thread tile
for (int j = 0; j < TM; j++) {
    for (int l = 0; l < TN; l++)
        tmp[j][l] += As[(ty + j) * BK + i] * Bs[tx + l + i * BN];
}
```

Global access to stock: Compared with the version without the introduction of shared memory cache, the global memory access to stock is reduced to `1/2*(1/BM+1/BN)=1/128`, and the access to stock is significantly reduced.

![](./images/kernel_3_vs_4.png)

The actual test found that compared with the one-dimensional thread tile, the matrix multiplication efficiency was significantly doubled because the two-dimensional thread tile further reduced the global access to stock and improved the computational access to memory ratio.

## kernel 5

**Register cache shared memory**

![](./images/describe_kernel_5.png)

As can be seen from the code below, a single thread calculates the thread tile element by multiplying and accumulating overtime, and the shared memory will be accessed repeatedly.

```cpp
for (int j = 0; j < TM; j++) {
    for (int l = 0; l < TN; l++)
        tmp[j][l]+=As[(ty+j)*BK+i]*Bs[tx+l+i*BN]; // In the inner loop, As[(ty+j)*BK+i] repeatedly accesses TN times
}
```

Shared memory can greatly reduce the memory access delay compared to global memory, but the shared memory delay (dozens of cycles) is still larger than the calculation delay (several cycles). Therefore, registers are used to cache the shared memory As and Bs to avoid repeated accesses to the shared memory.；

```cpp
float a_frag[TM] = {0.};

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N_DIM 28000            // Sized for RTX 4060 8GB VRAM (2 arrays, ~6.27GB total)
#define ITERATIONS 20000       // Enough iterations for diffusion to actually be visible
#define DT 0.01f
#define ALPHA 0.05f
#define SAVE_EVERY 200          // Save every 200 steps instead of every step (avoids I/O bottleneck)

#define TILE 32                 // Must match blockDim
#define SUB_N 512                // Observation window size, kept small so diffusion is visible

// ---------------------------------------------------------
// Error-check macro: catch CUDA call failures immediately
// ---------------------------------------------------------
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// ---------------------------------------------------------
// Shared-memory tiling kernel
// - Instead of each thread reading global memory 5 times,
//   the whole block loads a tile (with halo) into shared memory once and reuses it.
// - Fewer global memory accesses -> faster for this memory-bound kernel.
// ---------------------------------------------------------
__global__ void diffuse_tiled_kernel(const float* __restrict__ grid_in,
                                      float* __restrict__ grid_out, int N) {
    __shared__ float tile[TILE + 2][TILE + 2]; // +1 halo cell on each side

    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;

    int lx = threadIdx.x + 1;
    int ly = threadIdx.y + 1;

    if (row < N && col < N) {
        tile[ly][lx] = grid_in[row * N + col];

        // Halo fill: only edge threads of the tile also load the neighbor value
        if (threadIdx.x == 0 && col > 0)
            tile[ly][0] = grid_in[row * N + (col - 1)];
        if (threadIdx.x == TILE - 1 && col < N - 1)
            tile[ly][TILE + 1] = grid_in[row * N + (col + 1)];
        if (threadIdx.y == 0 && row > 0)
            tile[0][lx] = grid_in[(row - 1) * N + col];
        if (threadIdx.y == TILE - 1 && row < N - 1)
            tile[TILE + 1][lx] = grid_in[(row + 1) * N + col];
    }

    __syncthreads();

    if (row > 0 && row < N - 1 && col > 0 && col < N - 1) {
        float laplacian = tile[ly + 1][lx] + tile[ly - 1][lx] +
                           tile[ly][lx + 1] + tile[ly][lx - 1] -
                           4.0f * tile[ly][lx];
        grid_out[row * N + col] = tile[ly][lx] + ALPHA * laplacian * DT;
    }
}

// ---------------------------------------------------------
// Save sub-grid: one batched write to a text file
// ---------------------------------------------------------
void save_sub_grid(const float* h_sub, int sub_n, int step) {
    char filename[64];
    sprintf(filename, "extreme_step_%d.txt", step);
    FILE* fp = fopen(filename, "w");
    for (int i = 0; i < sub_n * sub_n; ++i) {
        fprintf(fp, "%.4f ", h_sub[i]);
        if ((i + 1) % sub_n == 0) fprintf(fp, "\n");
    }
    fclose(fp);
}

int main() {
    size_t size = (size_t)N_DIM * N_DIM * sizeof(float);
    printf("Allocating %.2f GB x2 on device...\n", size / 1e9);

    float *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMemset(d_a, 0, size));

    // Initial condition: small Gaussian blob instead of a single center point,
    // so diffusion is visible at a reasonable scale within ITERATIONS
    {
        int blob_r = 6;
        float* h_blob = (float*)malloc((2 * blob_r + 1) * (2 * blob_r + 1) * sizeof(float));
        int w = 2 * blob_r + 1;
        for (int dy = -blob_r; dy <= blob_r; ++dy) {
            for (int dx = -blob_r; dx <= blob_r; ++dx) {
                float d2 = (float)(dx * dx + dy * dy);
                h_blob[(dy + blob_r) * w + (dx + blob_r)] = 100.0f * expf(-d2 / (2.0f * 3.0f * 3.0f));
            }
        }
        for (int dy = -blob_r; dy <= blob_r; ++dy) {
            CUDA_CHECK(cudaMemcpy(
                &d_a[(N_DIM / 2 + dy) * N_DIM + (N_DIM / 2 - blob_r)],
                &h_blob[(dy + blob_r) * w],
                w * sizeof(float), cudaMemcpyHostToDevice));
        }
        free(h_blob);
    }

    dim3 blockSize(TILE, TILE);
    dim3 gridSize((N_DIM + TILE - 1) / TILE, (N_DIM + TILE - 1) / TILE);

    // ---------------------------------------------------------
    // Pinned memory: use cudaMallocHost instead of malloc
    // -> host<->device transfers go through page-locked memory,
    //    enabling direct DMA and faster copies than regular malloc.
    // ---------------------------------------------------------
    float* h_sub;
    CUDA_CHECK(cudaMallocHost(&h_sub, (size_t)SUB_N * SUB_N * sizeof(float)));

    // Stream for overlapping async transfer/execution
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    float *curr = d_a, *next = d_b;
    int save_count = 0;

    cudaEvent_t t_start, t_end;
    CUDA_CHECK(cudaEventCreate(&t_start));
    CUDA_CHECK(cudaEventCreate(&t_end));
    CUDA_CHECK(cudaEventRecord(t_start));

    for (int i = 0; i <= ITERATIONS; ++i) {
        if (i % SAVE_EVERY == 0) {
            // One batched 2D copy instead of looping cudaMemcpy 1024 times
            CUDA_CHECK(cudaMemcpy2DAsync(
                h_sub, SUB_N * sizeof(float),
                &curr[(N_DIM / 2 - SUB_N / 2) * N_DIM + (N_DIM / 2 - SUB_N / 2)],
                N_DIM * sizeof(float),
                SUB_N * sizeof(float), SUB_N,
                cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            save_sub_grid(h_sub, SUB_N, i);
            save_count++;
        }

        if (i < ITERATIONS) {
            diffuse_tiled_kernel<<<gridSize, blockSize, 0, stream>>>(curr, next, N_DIM);
            float* tmp = curr; curr = next; next = tmp;
        }
    }

    CUDA_CHECK(cudaEventRecord(t_end));
    CUDA_CHECK(cudaEventSynchronize(t_end));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t_start, t_end));

    printf("Simulation complete. Saved %d snapshots over %d iterations.\n", save_count, ITERATIONS);
    printf("Total GPU time: %.2f ms (%.2f ms/iteration)\n", ms, ms / ITERATIONS);

    cudaFreeHost(h_sub);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaStreamDestroy(stream);
    return 0;
}

#include <metal_stdlib>

using namespace metal;

#define TILE_SIZE 32

kernel void mm_naive(device float *mat1 [[buffer(0)]],
                     device float *mat2 [[buffer(1)]],
                     device float *mat3 [[buffer(2)]],
                     constant int *shape [[buffer(3)]],
                     uint3 global_idx [[thread_position_in_grid]]
) {
    unsigned int n = shape[0];
    unsigned int k = shape[1];
    unsigned int m = shape[2];
    
    if (global_idx.y < n && global_idx.x < m) {
        float sum = 0.0f;
        for (unsigned int i = 0; i < k; i++) {
            sum += mat1[global_idx.y * k + i] * mat2[i * m + global_idx.x];
        }
        mat3[global_idx.y * m + global_idx.x] = sum;
    }
}

kernel void mm_tiled(device float *mat1 [[buffer(0)]],
                     device float *mat2 [[buffer(1)]],
                     device float *mat3 [[buffer(2)]],
                     constant int *shape [[buffer(3)]],
                     uint3 global_idx [[thread_position_in_grid]],
                     uint3 threadIdx [[thread_position_in_threadgroup]]
                     ) {
    unsigned int N = shape[0];
    unsigned int K = shape[1];
    unsigned int M = shape[2];
    
    threadgroup float mat1_tile[TILE_SIZE][TILE_SIZE];
    threadgroup float mat2_tile[TILE_SIZE][TILE_SIZE];
    unsigned int num_phases = ((K + TILE_SIZE - 1) / TILE_SIZE);
    
    float sum = 0.0f;
    for (unsigned int phase = 0; phase < num_phases; phase++) {
        
        if (global_idx.y < N && (phase * TILE_SIZE + threadIdx.x) < K) {
            mat1_tile[threadIdx.y][threadIdx.x] = mat1[global_idx.y * K +  (phase * TILE_SIZE + threadIdx.x)];
        }
        else {
            mat1_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
                      
        if ((phase * TILE_SIZE + threadIdx.y) < K && global_idx.x < M) {
            mat2_tile[threadIdx.y][threadIdx.x] = mat2[(phase * TILE_SIZE + threadIdx.y) * M + global_idx.x];
        }
        else {
            mat2_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (unsigned int k = 0; k < TILE_SIZE; k++) {
            sum += mat1_tile[threadIdx.y][k] * mat2_tile[k][threadIdx.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (global_idx.y < N && global_idx.x < M) {
        mat3[global_idx.y * M + global_idx.x] = sum;
    }
}

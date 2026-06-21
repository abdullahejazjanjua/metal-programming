#include <metal_stdlib>

using namespace metal;

#define TILE_SIZE 32

kernel void mm_naive(device float *mat1 [[buffer(0)]],
                     device float *mat2 [[buffer(1)]],
                     device float *mat3 [[buffer(2)]],
                     constant int *shape [[buffer(3)]],
                     uint3 index [[thread_position_in_grid]]
) {
    unsigned int k = shape[1];
    unsigned int m = shape[2];
    
    float sum = 0.0f;
    for (unsigned int i = 0; i < k; i++) {
        sum += mat1[index.y * k + i] * mat2[i * m + index.x];
    }
    mat3[index.y * m + index.x] = sum;
}

kernel void mm_tiled(device float *mat1 [[buffer(0)]],
                     device float *mat2 [[buffer(1)]],
                     device float *mat3 [[buffer(2)]],
                     constant int *shape [[buffer(3)]],
                     uint3 index [[thread_position_in_grid]]
                     ) {
    unsigned int N = shape[0];
    unsigned int K = shape[1];
    unsigned int M = shape[2];
    
    threadgroup float mat1_tile[TILE_SIZE][TILE_SIZE];
    threadgroup float mat2_tile[TILE_SIZE][TILE_SIZE];
    unsigned int num_phases = ((K + TILE_SIZE - 1) / TILE_SIZE);
    
    float sum = 0.0f;
    for (unsigned int phase = 0; phase < num_phases; phase++) {
        
        if (index.y < N && (phase * TILE_SIZE + index.x) < K) {
            mat1_tile[index.y][index.x] = mat1[index.y * K +  (phase * TILE_SIZE) + index.x];
        }
        else {
            mat1_tile[index.y][index.x] = 0.0f;
        }
                      
        if ((phase * TILE_SIZE + index.y) < K && index.x < M) {
            mat2_tile[index.y][index.x] = mat2[(phase * TILE_SIZE + index.y) * M + index.x];
        }
        else {
            mat2_tile[index.y][index.x] = 0.0f;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (unsigned int k = 0; k < TILE_SIZE; k++) {
            sum += mat1_tile[index.y][k] * mat2_tile[k][index.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (index.y < N && index.x < M) {
        mat3[index.y * M + index.x] = sum;
    }
}

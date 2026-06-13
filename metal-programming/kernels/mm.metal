#include <metal_stdlib>
using namespace metal;

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

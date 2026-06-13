#include <metal_stdlib>
using namespace metal;

kernel void mm_naive(
    device float* mat1 [[buffer(0)]], // n, k
    device float* mat2 [[buffer(1)]], // k. m
    device float* mat3 [[buffer(2)]], // [n. k, m]
    constant unsigned int *size [[buffer(3)]],
    unsigned int index [[thread_position_in_grid]]
) {
    uint n = size[0];
    uint k = size[1];
    uint m = size[2];

    float sum = 0.0f;    
    for (int i = 0; i < k; i++) {
        sum += mat1[index.y * k + i] * mat2[i * m + index.x];
    }
    mat3[index.y * m + index.x] = sum;
}
#include <metal_stdlib>
using namespace metal;

// The [[]] in MSL is attribute qualifiers, it tells the metal compiler what to put in the variable on launch
// Resource Binding: [[buffer(i)]] tells which memory location in RAM to map to this
// Built-in Variables: [[thread_position_in_grid]] Commands the GPU hardware to inject internal system values (like the current thread's index) directly into that variable upon launch

kernel void vector_add(device float *A [[buffer(0)]],
                       device float *B [[buffer(1)]],
                       device float *C [[buffer(2)]],
                       uint3 pos [[thread_position_in_grid]]
) {
    C[pos.x] = A[pos.x] + B[pos.x];
}

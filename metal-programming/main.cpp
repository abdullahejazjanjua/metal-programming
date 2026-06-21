#include <Metal/Metal.hpp>
#include "include/mm.hpp"

#define N 1000
#define K 5000
#define M 1000

int main() {
    MTL::Device *device = MTL::CreateSystemDefaultDevice();
    unsigned int shape[] = {N, K, M};
    
    mm mm_kernel(device, "mm_tiled");
    
//    mm_kernel.debugger();
//    mm_kernel.start_debugger();
    
    mm_kernel.allocate_buffers(shape);
    mm_kernel.launch_kernel(32);
    
//    mm_kernel.stop_debugger();
    mm_kernel.verify_results();
    mm_kernel.print();
}

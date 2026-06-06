#include <iostream>
#include <Metal/Metal.hpp>

#define N 10000000

int main(int argc, const char * argv[]) {
    
//    std::vector<float *> A();
    
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (device == nullptr) {
        throw std::runtime_error("Metal is not supported on this device\n");
    }
    MTL::CommandQueue* cmd_queue = device->newCommandQueue();
    if (cmd_queue == nullptr) {
        device->release();
        throw std::runtime_error("Couldn't create a commmand queue\n");
    }
    
    // When you build your project, Xcode automatically finds all files with the .metal extension within your target. It compiles them and combines them into a single binary archive file named default.metallib. This file is then placed inside your final application bundle.
    // When you call device->newDefaultLibrary(), you are instructing the GPU to search the application bundle, locate default.metallib, and load all of your compiled kernel functions into memory so they can be executed.
    
    // retrieve the compiled library
    MTL::Library* library = device->newDefaultLibrary();
    
    // Extract the kernel function by its exact name in the .metal file
    NS::String* func_name = NS::String::string("vector_add", NS::UTF8StringEncoding);
    MTL::Function* func = library->newFunction(func_name); // this retrieves in metal asm, we need machine code
    
    // compile the metal asm into machine code
    NS::Error* error = nullptr;
    MTL::ComputePipelineState* pipeline = device->newComputePipelineState(func, &error);
    
    // Create buffers
    uint size = N * sizeof(float);
    
    // Allocate buffers but both GPU/CPU can read/write
    MTL::Buffer* bufferA = device->newBuffer(size, MTL::ResourceStorageModeShared);
    MTL::Buffer* bufferB = device->newBuffer(size, MTL::ResourceStorageModeShared);
    MTL::Buffer* bufferC = device->newBuffer(size, MTL::ResourceStorageModeShared);
    
    float *A = (float*) bufferA->contents();
    float *B = (float*) bufferB->contents();
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }
    
    // Create a buffer that holds commands to be executed by the GPU
    MTL::CommandBuffer* cmd_buffer = cmd_queue->commandBuffer();
    MTL::ComputeCommandEncoder* compute_enc = cmd_buffer->computeCommandEncoder();
    
    // This maps to passing arguments inside the CUDA <<< >>> execution configuration. You bind the pipeline state, and then assign the allocated buffers to the specific indices expected by the [[buffer(n)]] attributes in your .metal file.
    compute_enc->setComputePipelineState(pipeline);
    compute_enc->setBuffer(bufferA, 0, 0);
    compute_enc->setBuffer(bufferB, 0, 1);
    compute_enc->setBuffer(bufferC, 0, 2);
    
    // This defines the grid and block dimensions. Metal handles boundary checks internally when using dispatchThreads, so you provide the exact total number of elements ($N$) rather than calculating the number of blocks manually.
    
    MTL::Size threadsPerThreadgroup = MTL::Size(32, 1, 1); // block
    MTL::Size threadsPerGrid = MTL::Size(N, 1, 1); // grid
    
    compute_enc->dispatchThreads(threadsPerGrid, threadsPerThreadgroup);
    
    // cleanup
    compute_enc->endEncoding();
    cmd_buffer->commit();
    cmd_buffer->waitUntilCompleted();
    
    float *C = (float*) bufferC->contents();
    for (int i = 0; i < 10; i++) {
        std :: cout << C[i] << " ";
    }
    std :: cout << "\n";
    
    bufferA->release();
    bufferB->release();
    bufferC->release();
    pipeline->release();
    cmd_queue->release();
    device->release();
    
    return 0;
}

#include "include/mm.hpp"
#include <iostream>
#include <cmath>

#define cdiv(val, div) ((val + div - 1) / (div))


mm::mm(MTL::Device* device_ptr, std::string shader_name) {
    this->device_ptr = device_ptr;
    make_shader(shader_name);
}

void mm::make_shader(std::string shader_name) {
    this->library_ptr = this->device_ptr->newDefaultLibrary();
    if (this->library_ptr == nil) {
        std :: cerr << "Failed to retrieve the default library containing containing .metal\n";
    }
    
    auto shader_name_ns = NS::String::string(shader_name.c_str(), NS::ASCIIStringEncoding);
    this->func_ptr = this->library_ptr->newFunction(shader_name_ns);
    if (this->func_ptr == nil) {
        std :: cerr << "Failed to retrieve shader " << shader_name << " from the default library\n";
    }
    
    NS::Error* err;
    this->pipeline_state = this->device_ptr->newComputePipelineState(this->func_ptr, &err);
    if (this->pipeline_state == nil) {
        std :: cerr << "Failed to create a pipeline state object\n";
    }
    
    this->cmd_queue = this->device_ptr->newCommandQueue();
    if (this->cmd_queue == nil) {
        std :: cerr << "Failed to create a command queue\n";
    }
    
}

void mm::allocate_buffers(unsigned int *shape) {
    int n = shape[0];
    int k = shape[1];
    int m = shape[2];
    
    this->bufferMat1 = this->device_ptr->newBuffer((n*k) * sizeof(float), MTL::ResourceStorageModeShared);
    this->bufferMat2 = this->device_ptr->newBuffer((k*m) * sizeof(float), MTL::ResourceStorageModeShared);
    this->bufferMat3 = this->device_ptr->newBuffer((n*m) * sizeof(float), MTL::ResourceStorageModeShared);
    
    this->bufferShape = this->device_ptr->newBuffer(shape, 3 * sizeof(int), MTL::ResourceStorageModeShared);
    
    this->generateRandomFloatData(bufferMat1);
    this->generateRandomFloatData(bufferMat2);
    this->generateRandomFloatData(bufferMat3);
}

void mm::generateRandomFloatData(MTL::Buffer* buf) {
    float *data = static_cast<float*>(buf->contents());
    
    for (unsigned long i = 0; i < buf->length() / sizeof(float); i++) {
        data[i] = (float)rand() / (float)(RAND_MAX);
    }
}
    
void mm::launch_kernel(int num_threads) {
    int *shape = static_cast<int*>(this->bufferShape->contents());
    int n = shape[0];
    int m = shape[2];
    
    MTL::CommandBuffer* cmd_buffer = this->cmd_queue->commandBuffer();
    MTL::ComputeCommandEncoder* compute_encoder = cmd_buffer->computeCommandEncoder();
    
    compute_encoder->setComputePipelineState(this->pipeline_state);
    compute_encoder->setBuffer(this->bufferMat1, 0, 0);
    compute_encoder->setBuffer(this->bufferMat2, 0, 1);
    compute_encoder->setBuffer(this->bufferMat3, 0, 2);
    compute_encoder->setBuffer(this->bufferShape, 0, 3);
    
    MTL::Size gridSize = MTL::Size::Make(m, n, 1);
    MTL::Size threadgroupSize = MTL::Size::Make(num_threads, num_threads, 1);
    
    compute_encoder->dispatchThreads(gridSize, threadgroupSize);
    compute_encoder->endEncoding();
    
    cmd_buffer->commit();
    cmd_buffer->waitUntilCompleted();
}

void mm::debugger() {
//    NS::AutoreleasePool* mem_pool = NS::AutoreleasePool::alloc()->init();
    
    this->cap_mag = MTL::CaptureManager::sharedCaptureManager();
    this->cap_desc = MTL::CaptureDescriptor::alloc()->init();;
    this->cap_desc->setCaptureObject(this->device_ptr);
//    this->cap_desc->setDestination(MTL::CaptureDestinationGPUTraceDocument);
    
    
}

void mm::start_debugger() {
    NS::Error* err;
    if (!this->cap_mag->startCapture(this->cap_desc, &err)) {
        std :: cerr << "Error: " << err->localizedDescription()->utf8String();
    }
}

void mm::stop_debugger() {
    this->cap_mag->stopCapture();
    this->cap_desc->release();
    this->cap_mag->release();
}


void mm::print() {
    float *data = static_cast<float*>(this->bufferMat3->contents());
    for (int i = 0; i < 10; i++) {
        std :: cout << "data[" << i << "] = " << data[i] << "\n";
    }
}

void mm::verify_results() {
    float *mat1 = static_cast<float*>(this->bufferMat1->contents());
    float *mat2 = static_cast<float*>(this->bufferMat2->contents());
    float *mat3 = static_cast<float*>(this->bufferMat3->contents());
    
    unsigned int *shape = static_cast<unsigned int*>(this->bufferShape->contents());
    
    for (int i = 0; i < shape[0]; i++) { // N
        for (int j = 0; j < shape[2]; j++) { // M
            float sum = 0.0f;
            for (int k = 0; k < shape[1]; k++) { // K
                 sum += mat1[i * shape[1] + k] * mat2[k * shape[2] + j];
            }
            float diff = std::abs(mat3[i * shape[2] + j] - sum);
            if (diff >  1e-4f)
                std::cout << "Mismatch at [" << i << "," << j << "]! " << "GPU: " << mat3[i * shape[2] + j] << ", CPU: " << sum << ", Diff: " << diff << "\n";
        }
    }

}

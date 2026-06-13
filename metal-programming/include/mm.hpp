#ifndef MM_HPP
#define MM_HPP

#include <Metal/Metal.hpp>

class mm {
    
public:
    mm(MTL::Device* device_ptr, std::string shader_name);
    void make_shader(std::string shader_name);
    void allocate_buffers(unsigned int *shape);
    void generateRandomFloatData(MTL::Buffer* buf);
    void launch_kernel(int num_threads);
    void print();
    void verify_results();
    
private:
    MTL::Device *device_ptr;
    MTL::Library* library_ptr;
    MTL::Function* func_ptr;
    MTL::ComputePipelineState* pipeline_state;
    MTL::CommandQueue* cmd_queue;
    
    MTL::Buffer* bufferMat1;
    MTL::Buffer* bufferMat2;
    MTL::Buffer* bufferMat3;
    
    MTL::Buffer* bufferShape;
};


#endif

cmake_minimum_required(VERSION 3.15)
project(MetalProgramming CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)


add_executable(metal-programming 
    main.cpp 
    mtl_implementation.cpp
)

target_include_directories(metal-programming PRIVATE 
    /Users/abdullahejazjanjua/Library/metal-cpp
)

target_link_libraries(metal-programming PRIVATE 
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${QUARTZCORE_FRAMEWORK}
)

add_custom_command(
    TARGET metal-programming POST_BUILD
    COMMAND xcrun -sdk macosx metal -c ${CMAKE_CURRENT_SOURCE_DIR}/vecAdd-kernel.metal -o ${CMAKE_CURRENT_BINARY_DIR}/vecAdd-kernel.air
    COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/vecAdd-kernel.air -o ${CMAKE_CURRENT_BINARY_DIR}/default.metallib
    COMMENT "Compiling Metal Shaders into default.metallib..."
)
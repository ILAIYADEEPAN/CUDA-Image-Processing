
cmake_minimum_required(VERSION 3.18)
project(GPUImageProcessing CUDA CXX)

find_package(OpenCV REQUIRED)

add_executable(gpu_image_processing src/main.cu)

target_include_directories(gpu_image_processing PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(gpu_image_processing ${OpenCV_LIBS})

set_target_properties(gpu_image_processing PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

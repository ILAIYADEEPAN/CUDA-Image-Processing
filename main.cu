
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

__global__ void grayscaleKernel(unsigned char* input,
                                unsigned char* output,
                                int width,
                                int height,
                                int channels) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {

        int idx = (y * width + x) * channels;

        unsigned char b = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char r = input[idx + 2];

        output[y * width + x] =
            (r + g + b) / 3;
    }
}

void processImage(const std::string& inputPath,
                  const std::string& outputPath) {

    cv::Mat img = cv::imread(inputPath);

    if (img.empty()) {
        std::cout << "Failed to load image: " << inputPath << std::endl;
        return;
    }

    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();

    int inputSize = width * height * channels;
    int outputSize = width * height;

    unsigned char *d_input, *d_output;

    cudaMalloc((void**)&d_input, inputSize);
    cudaMalloc((void**)&d_output, outputSize);

    cudaMemcpy(d_input,
               img.data,
               inputSize,
               cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16,
                (height + 15) / 16);

    grayscaleKernel<<<blocks, threads>>>(
        d_input,
        d_output,
        width,
        height,
        channels
    );

    cudaDeviceSynchronize();

    cv::Mat gray(height, width, CV_8UC1);

    cudaMemcpy(gray.data,
               d_output,
               outputSize,
               cudaMemcpyDeviceToHost);

    cv::imwrite(outputPath, gray);

    cudaFree(d_input);
    cudaFree(d_output);

    std::cout << "Processed: " << inputPath << std::endl;
}

int main() {

    std::string inputFolder = "../images/";
    std::string outputFolder = "../output/";

    for (const auto& entry : fs::directory_iterator(inputFolder)) {

        std::string inputPath = entry.path().string();

        std::string filename = entry.path().filename().string();

        std::string outputPath =
            outputFolder + "gray_" + filename;

        processImage(inputPath, outputPath);
    }

    std::cout << "Batch Processing Completed!" << std::endl;

    return 0;
}

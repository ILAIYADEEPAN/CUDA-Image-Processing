
# GPU Accelerated Batch Image Processing Using CUDA

## Features
- Batch image processing
- GPU accelerated grayscale conversion
- CUDA parallel processing
- OpenCV image handling

---

## Requirements
- NVIDIA GPU
- CUDA Toolkit
- OpenCV
- CMake
- Visual Studio / GCC

---

## Project Structure

GPU-Image-Processing/
│
├── images/
├── output/
├── src/
│   └── main.cu
│
├── CMakeLists.txt
└── README.md

---

## Build Instructions

### Step 1
Install:
- CUDA Toolkit
- OpenCV
- CMake

### Step 2

Create build folder:

mkdir build
cd build

### Step 3

Run:

cmake ..

### Step 4

Compile:

cmake --build .

### Step 5

Place input images inside:
images/

### Step 6

Run executable.

Processed grayscale images will appear in:
output/

---

## CUDA Concepts Used
- CUDA kernels
- Threads and blocks
- GPU memory allocation
- Parallel image processing

---

## Output
Each image is converted into grayscale using GPU acceleration.

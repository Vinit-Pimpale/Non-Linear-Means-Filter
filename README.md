# Non-Local Means (NLM) Image Denoising – CPU and GPU Implementations

This repository contains a complete and reproducible implementation of the **Non-Local Means (NLM)** image denoising algorithm on both **CPU** and **GPU**, developed as part of the Winter Semester 2023 coursework at the University of Stuttgart.

The project focuses on faithful algorithmic implementation, performance comparison between CPU and GPU, and the application of professional software engineering practices suitable for scientific and high-performance computing.

This README serves as a technical appendix to the submitted project report.

---

## 1. Project Context and Objectives

The Non-Local Means (NLM) filter is a patch-based image denoising algorithm that exploits redundancy across the entire image domain. Unlike local filters, NLM preserves edges and fine textures by comparing pixel neighborhoods instead of relying solely on spatial proximity.

The objectives of this project were:

1. Implement the NLM filter on CPU (custom implementation).
2. Implement the NLM filter on GPU using OpenCL.
3. Compare execution performance with and without memory transfer overhead.
4. Validate correctness through output consistency.
5. Apply clean, maintainable, and reproducible coding practices.

---

## 2. Algorithm Overview

For each pixel at location (x₀, y₀):

1. Extract a local patch of size P × P.
2. Search similar patches within a search window of size N × N.
3. Compute Euclidean distances between patches.
4. Convert distances to weights using a Gaussian kernel controlled by parameter h.
5. Compute the filtered pixel as a normalized weighted average.

The parameter h controls the trade-off between noise reduction and detail preservation.

---

## 3. Input Data and Parameters

### Input Image
- Resolution: 512 × 512
- Format: Grayscale PGM
- Noise model: Gaussian noise
  - Mean: 0.005
  - Variance: 0.005

### Parameters
- Patch size: 7 × 7
- Search window: 21 × 21
- Filtering strength:
  - Custom CPU: h = 200
  - OpenCV reference: h = 15

---

## 4. CPU Implementations

### 4.1 OpenCV Reference Implementation

The OpenCV function `cv::fastNlMeansDenoising` was used as a correctness and performance reference.

### 4.2 Custom CPU Implementation

The custom CPU implementation explicitly performs:

- Nested iteration over image pixels
- Search window traversal
- Patch-wise distance computation
- Gaussian weight calculation
- Normalized weighted summation

This implementation prioritizes clarity and correctness and serves as a baseline for parallelization.

---

## 5. GPU Implementation (OpenCL)

The GPU implementation assigns one work-item per output pixel.

Key aspects:
- Boundary-safe memory access
- Identical algorithmic logic to CPU
- OpenCL kernel-based parallelism
- Minimal host-device memory overhead

The GPU version preserves algorithmic equivalence with the CPU implementation.

---

## 6. Performance Evaluation

### Hardware
- CPU: Intel Xeon E5-2620 @ 2.00 GHz
- GPU: NVIDIA GeForce GTX 680 (GK104)
- System: kale.cis.iti.uni-stuttgart.de

### Timing Results

- CPU Time: 12.98 s (0.020 MPixel/s)
- GPU Time (kernel only): 0.031 s (8.33 MPixel/s)
- GPU Time (with memory copy): 0.032 s (8.24 MPixel/s)

The GPU achieves a speedup exceeding 400× over the CPU baseline.

---

## 7. Repository Structure

```text
Non-Linear-Means-Filter/
├── src/                # CPU and GPU implementations
├── lib/                # Helper utilities
├── input_img/          # Input images
├── output_img/         # Filtered outputs
├── meson.build         # Build configuration
├── .clang-format       # Code style rules
├── .gitignore          # Version control hygiene
└── README.md           # Documentation
```
---

## 8. Build Instructions

### Requirements
- C++17 compatible compiler
- OpenCL runtime
- Meson >= 0.60
- Ninja (recommended)

### Build
meson setup build
meson compile -C build

---

## 9. Software Engineering Practices

This project demonstrates:
- Deterministic execution
- Clear separation of concerns
- Explicit parameter definitions
- Maintainable code structure
- Reproducible builds

---

## 10. Limitations and Future Work

Future improvements may include:
- OpenMP CPU parallelization
- GPU shared-memory optimization
- PSNR and SSIM evaluation
- Support for color images

---

## 11. Authors

Vinit Pimpale  
M.Sc. Information Technology  
University of Stuttgart  

Antara Dey  
University of Stuttgart  

---

## 12. References

1. Buades, A., Coll, B., and Morel, J.-M., A Non-Local Algorithm for Image Denoising, CVPR 2005.
2. Wang et al., An Improved Non-Local Means Filter for Color Image Denoising, Optik 2018.
3. OpenCV Documentation – Denoising
4. IPOL – Image Processing On Line

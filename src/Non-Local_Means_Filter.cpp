#include <opencv2/opencv.hpp>
#include <stdio.h>

#include <Core/Assert.hpp>
#include <Core/Image.hpp>
#include <Core/Time.hpp>
#include <OpenCL/Device.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/cl-patched.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

/* Non-Local Means filter CPU implementation
 * @input Input image
 * @h Patch similarity parameter
 * @searchSize search window size, in this case 21
 * @patchSize patch size for comparison of pixels, in this case 7
 * @result cv::Mat filtered Image returned to calling function
 */

cv::Mat nlmFilter(const cv::Mat& input, double h, int searchSize, int patchSize) {
    // Initialize the result image object
    cv::Mat result = cv::Mat::zeros(input.size(), input.type());

    int halfPatch = patchSize / 2; // Half size of the patch
    int halfSearch = searchSize / 2; // Half size of the search window

    // Iterate over each pixel of the input image
    for (int y = halfSearch; y < input.rows - halfSearch; ++y) {
        for (int x = halfSearch; x < input.cols - halfSearch; ++x) {
            // Initialize variables for weighted sum and filtered pixel value
            double weightSum = 0.0;
            double filteredPixel = 0.0;

            // Iterate over the search window
            for (int i = -halfSearch; i <= halfSearch; ++i) {
                for (int j = -halfSearch; j <= halfSearch; ++j) {
                    // Initialize sum for patch difference
                    double patchSum = 0.0;
                    // Iterate through the pixel neighbours/over the patch region
                    for (int m = -halfPatch; m <= halfPatch; ++m) {
                        for (int n = -halfPatch; n <= halfPatch; ++n) {
                            // Calculate pixel intensity differences between patches
                            double diff = static_cast<double>(input.at<uchar>(y + m, x + n) -
                                                              input.at<uchar>(y + i + m, x + j + n));
                            patchSum += diff * diff; // Accumulate squared differences
                        }
                    }

                    double euclideanDistance = sqrt(patchSum); // Calculate Euclidean distance between neigboring pixels
                    double weight = exp (- ( patchSum * euclideanDistance ) / (h * h) ); // Calculate weight

                    // Accumulate weighted sum for filtered pixel
                    filteredPixel += weight * static_cast<double>(input.at<uchar>(y + i, x + j));
                    weightSum += weight;
                }
            }

            // Compute the final filtered pixel value
            result.at<uchar>(y, x) = static_cast<uchar>(filteredPixel / weightSum);

            // Debug print statement
            //std::cout << "CPU----Pixel at (" << x << ", " << y << "): " << static_cast<int>(result.at<uchar>(y, x)) << std::endl;
        }
    }
    // Return the filtered image
    return result;
}

int main(int argc, char** argv) {
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Non-Local Means Filter" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // Declare some values
    //std::size_t wgSizeX = 4;  // Number of work items per work group in X direction (128 px)
    //std::size_t wgSizeY = 4;
    std::size_t wgSizeX = 16;  // Number of work items per work group in X direction (512 px)
    std::size_t wgSizeY = 16;
    std::size_t countX = wgSizeX * 32;  // Overall number of work items in X direction = Number of elements in X direction
    std::size_t countY = wgSizeY * 32;

    std::size_t count = countX * countY;       // Overall number of elements
    std::size_t size = count * sizeof(float);  // Size of data in bytes

    // Allocate space for output data from CPU and GPU on the host
    std::vector<float> h_input(count);
    std::vector<float> h_outputCpu(count);
    std::vector<float> h_outputGpu(count);

    /*----------------------------------------------------------------
    * NLM filter for GPU
    *----------------------------------------------------------------*/

    // Create a context
    cl::Context context(CL_DEVICE_TYPE_GPU);

    // Get a device of the context
    int deviceNr = argc < 2 ? 1 : atoi(argv[1]);
    std::cout << "Using device " << deviceNr << " / "
            << context.getInfo<CL_CONTEXT_DEVICES>().size() << std::endl;
    ASSERT(deviceNr > 0);
    ASSERT((size_t)deviceNr <= context.getInfo<CL_CONTEXT_DEVICES>().size());
    cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[deviceNr - 1];
    std::vector<cl::Device> devices;
    devices.push_back(device);
    OpenCL::printDeviceInfo(std::cout, device);

    // Create a command queue
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    // Load the source code
    extern unsigned char Non_Local_Means_Filter_cl[];
    extern unsigned int Non_Local_Means_Filter_cl_len;
    cl::Program program(context,
                      std::string((const char*)Non_Local_Means_Filter_cl,
                                  Non_Local_Means_Filter_cl_len));
    // Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
    OpenCL::buildProgram(program, devices);

    // Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
    memset(h_input.data(), 255, size);
    memset(h_outputCpu.data(), 255, size);
    memset(h_outputGpu.data(), 255, size);

    // Reinitialize output memory to 0xff
    memset(h_outputGpu.data(), 255, size);


    /*----------------------------------------------------------------
    * NLM filter for CPU
    *----------------------------------------------------------------*/

    // Read Input Image for CPU
    //cv::Mat input_image = cv::imread("../input_img/einstein_128.pgm", cv::IMREAD_GRAYSCALE);
    cv::Mat input_image = cv::imread("../input_img/girl_512.pgm", cv::IMREAD_GRAYSCALE);

    if (input_image.empty()) {
        std::cerr << "Error: Unable to read the image." << std::endl;
        return -1;
    }

    // Convert image to grayscale if it's not already
    if (input_image.channels() > 1) {
        cv::cvtColor(input_image, input_image, cv::COLOR_BGR2GRAY);
    }

    // Adjust the parameters according to your needs
    int patchSize = 7;  // Size of the patch for comparison
    int searchSize = 21; // Size of the search window
    double h1 = 15;   // Controls the filtering strength
    double h2 = 600;   // Controls the filtering strength

    // Use an image (girl_512.pgm) as input data
    std::vector<float> inputData;
    std::size_t inputWidth, inputHeight;
    //Core::readImagePGM("../input_img/einstein_128.pgm", inputData, inputWidth, inputHeight);
    Core::readImagePGM("../input_img/girl_512.pgm", inputData, inputWidth, inputHeight);
    for (size_t j = 0; j < countY; j++) {
      for (size_t i = 0; i < countX; i++) {
        h_input[i + countX * j] =
            inputData[(i % inputWidth) + inputWidth * (j % inputHeight)];
      }
    }

    // Execute NLM filter by OpenCV library for verification
    cv::Mat output_image_opencv;
    cv::fastNlMeansDenoising (input_image, output_image_opencv, h1, searchSize, patchSize);

    // Execute NLM filter for CPU (self-implemented)
    Core::TimeSpan cpuStart = Core::getCurrentTime();
    cv::Mat output_image = nlmFilter(input_image, h2, searchSize, patchSize);
    Core::TimeSpan cpuEnd = Core::getCurrentTime();


    // Generate output images as files
    cv::imwrite("../output_img/girl_512_cpu.pgm", output_image);
    cv::imwrite("../output_img/girl_512_cpu_opencv.pgm", output_image_opencv);
    //cv::imwrite("../output_img/einstein_128_cpu.pgm", output_image);

    //////// Store CPU output image (to be used for output comparison) ///////////////////////////////////
    Core::readImagePGM("../output_img/girl_512_cpu.pgm", h_outputCpu, countX, countY);
    //Core::readImagePGM("../output_img/einstein_128_cpu.pgm", h_outputCpu, countX, countY);

    /*----------------------------------------------------------------
    * NLM filter for GPU
    *----------------------------------------------------------------*/

    // Initialize image variables
    cl::Event copy1;
    cl::Image2D inp_imageGpu;
    cl::Image2D out_imageGpu;
    inp_imageGpu = cl::Image2D(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_R, CL_FLOAT), countX, countY);
    out_imageGpu = cl::Image2D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), countX, countY);

    cl::size_t<3> origin;
    origin[0] = origin[1] = origin[2] = 0;
    cl::size_t<3> region;
    region[0] = countX;
    region[1] = countY;
    region[2] = 1;

    // Copy input data to input image object
    queue.enqueueWriteImage(inp_imageGpu, true, origin, region, 0, 0, h_input.data(), NULL, &copy1);

    // Create a kernel object
    cl::Kernel nlmFilter_Kernel(program, "nlmFilter_Kernel");

    // Launch kernel on the device
    cl::Event execution;
    nlmFilter_Kernel.setArg<cl::Image2D>(0, inp_imageGpu); // Input image
    nlmFilter_Kernel.setArg<cl::Image2D>(1, out_imageGpu); // Output image
    nlmFilter_Kernel.setArg<cl_uint>(2, countX);
    nlmFilter_Kernel.setArg<cl_uint>(3, countY);
    nlmFilter_Kernel.setArg<cl_double>(4, h2); // use h1 or h2

    queue.enqueueNDRangeKernel(nlmFilter_Kernel, cl::NullRange, cl::NDRange(countX, countY),
                               cl::NDRange(wgSizeX, wgSizeY), NULL, &execution);

    // Copy output data back to host
    cl::Event copy2;
    queue.enqueueReadImage(out_imageGpu, true, origin, region,
                           countX * sizeof(float), 0, h_outputGpu.data(), NULL, &copy2);

    // Convert result data to OpenCV Mat
    cv::Mat outputMat_Gpu(input_image.rows, input_image.cols, CV_8UC1, h_outputGpu.data());

    // Save the result as a .pgm file
    cv::imwrite("../output_img/girl_512_gpu.pgm", outputMat_Gpu);
    //cv::imwrite("../output_img/einstein_128_gpu.pgm", outputMat_Gpu);

    // Evaluate and print performance data
    Core::TimeSpan cpuTime = cpuEnd - cpuStart;
    Core::TimeSpan gpuTime = OpenCL::getElapsedTime(execution);
    Core::TimeSpan copyTime = OpenCL::getElapsedTime(copy1) + OpenCL::getElapsedTime(copy2);
    Core::TimeSpan overallGpuTime = gpuTime + copyTime;
    std::cout << "CPU Time: " << cpuTime.toString() << ", "
              << (count / cpuTime.getSeconds() / 1e6) << " MPixel/s"
              << std::endl;

    std::cout << "Memory copy Time: " << copyTime.toString() << std::endl;
    std::cout << "GPU Time w/o memory copy: " << gpuTime.toString()
              << " (speedup = " << (cpuTime.getSeconds() / gpuTime.getSeconds())
              << ", " << (count / gpuTime.getSeconds() / 1e6) << " MPixel/s)"
              << std::endl;
    std::cout << "GPU Time with memory copy: " << overallGpuTime.toString()
              << " (speedup = "
              << (cpuTime.getSeconds() / overallGpuTime.getSeconds()) << ", "
              << (count / overallGpuTime.getSeconds() / 1e6) << " MPixel/s)"
              << std::endl;

    // Check whether results are correct
    std::size_t errorCount = 0;
    for (size_t i = 0; i < countX; i = i + 1) {    //loop in the x-direction
      for (size_t j = 0; j < countY; j = j + 1) {  //loop in the y-direction
        size_t index = i + j * countX;
        // Allow small differences between CPU and GPU results (due to different rounding behavior)
        if (!(std::abs(h_outputCpu[index] - h_outputGpu[index]) <= 1e-5)) {
          if (errorCount < 15)
            std::cout << "Result for " << i << "," << j
                      << " is incorrect: GPU value is " << h_outputGpu[index]
                      << ", CPU value is " << h_outputCpu[index] << std::endl;
          else if (errorCount == 15)
            std::cout << "..." << std::endl;
          errorCount++;
        }
      }
    }
    if (errorCount != 0) {
      std::cout << "Found " << errorCount << " incorrect results" << std::endl;
      return 1;
    }

    std::cout << std::endl;
    std::cout << "Success" << std::endl;

    return 0;
}

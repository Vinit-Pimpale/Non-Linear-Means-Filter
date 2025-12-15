// Constant sampler with required properties
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE;

/* GPU Kernel function for the NLM (Non-Local Means) filter
 * @inputImage input image passed to kernel from host side
 * @resultImage Assign filtered image data, which will be copied back to host
 * @width Width of the image, in this case 512
 * @height Height of the image, in this case 512
 * @h Patch similarity parameter or filtereing strength
 */
__kernel void nlmFilter_Kernel(__read_only image2d_t inputImage,
                                write_only image2d_t resultImage,
                                uint width,
                                uint height,
                                double h) {       // Parameter for filtering strength

    // Get the global indices
    int countX = get_global_id(0);
    int countY = get_global_id(1);

    int patchSize = 3;  // Size of the patch for comparison (einstein.pgm)
    int searchSize = 11; // Size of the search window

    int halfPatch = patchSize / 2;
    int halfSearch = searchSize / 2;

    // Check if the current pixel is within the valid region; if not, exit the kernel
    if (countX < halfSearch || countX >= width - halfSearch ||
        countY < halfSearch || countY >= height - halfSearch) {
        return;
    }

    // Initialize variables for weighted sum and filtered pixel value
    double weightSum = 0.0;
    double filteredPixel = 0.0;

    // Iterate over the search window
    for (int i = -halfSearch; i <= halfSearch; ++i) {
        for (int j = -halfSearch; j <= halfSearch; ++j) {
            // Initialize sum for patch difference
            double patchSum = 0.0;
            // Iterate over the patch region
            for (int m = -halfPatch; m <= halfPatch; ++m) {
                for (int n = -halfPatch; n <= halfPatch; ++n) {
                    // Calculate coordinates for current and reference pixels
                    int2 currentPixelCoord = (int2)(countX + n, countY + m);
                    int2 referencePixelCoord = (int2)(countX + j + n, countY + i + m);

                    // Read pixel values from input image
                    uint4 currentPixel = read_imageui(inputImage, sampler, currentPixelCoord);
                    uint4 referencePixel = read_imageui(inputImage, sampler, referencePixelCoord);

                    // Calculate squared difference and accumulate patchSum
                    double diff = (double)(currentPixel.x) - (double)(referencePixel.x);
                    patchSum += diff * diff;
                }
            }

            double euclideanDistance = sqrt(patchSum); // Calculate Euclidean distance between neigboring pixels
            double weight = exp (- ( patchSum * euclideanDistance ) / (h * h) ); // Calculate weight

            // Weighted sum for filtered pixel
            filteredPixel += weight * (double)(read_imageui(inputImage, sampler, (int2)(countX + j, countY + i)).x);
            weightSum += weight;
        }
    }

    // Evaluate the final result pixel value
    uint4 resultPixel = (uint4)(filteredPixel / weightSum);

    // Debug print statement
    //printf("GPU-----Pixel at (%d, %d): %d\n", countX, countY, resultPixel.x);

    // Write the resultant pixel to the output image
    write_imageui(resultImage, (int2)(countX, countY), resultPixel);
}



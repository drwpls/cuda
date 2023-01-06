#include <stdio.h>
#include <stdint.h>
#include <vector_functions.h>

#define WIDTH_REMOVE 16

#define FILTER_WIDTH 3
// __constant__ float dc_filter[FILTER_WIDTH * FILTER_WIDTH];

#define CHECK(call)                                                \
    {                                                              \
        const cudaError_t error = call;                            \
        if (error != cudaSuccess)                                  \
        {                                                          \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,       \
                    cudaGetErrorString(error));                    \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    }

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

void readPnm(char *fileName, int &width, int &height, uchar3 *&pixels)
{
    FILE *f = fopen(fileName, "r");
    if (f == NULL)
    {
        printf("Cannot read %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    char type[3];
    fscanf(f, "%s", type);

    if (strcmp(type, "P3") != 0) // In this exercise, we don't touch other types
    {
        fclose(f);
        printf("Cannot read %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    fscanf(f, "%i", &width);
    fscanf(f, "%i", &height);

    int max_val;
    fscanf(f, "%i", &max_val);
    if (max_val > 255) // In this exercise, we assume 1 byte per value
    {
        fclose(f);
        printf("Cannot read %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
    for (int i = 0; i < width * height; i++)
        fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

    fclose(f);
}

void writePnm(uchar3 *pixels, int width, int height, char *fileName)
{
    FILE *f = fopen(fileName, "w");
    if (f == NULL)
    {
        printf("Cannot write %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    fprintf(f, "P3\n%i\n%i\n255\n", width, height);

    printf("\nImage output size (width x height): %i x %i\n", width, height);

    for (int i = 0; i < width * height; i++)
        fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);

    fclose(f);
}

void convertGrayscale(uchar3 *inPixels, int width, int height, uint8_t * grayPixels)
{
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int i = r * width + c;
            uint8_t red = inPixels[i].x;
            uint8_t green = inPixels[i].y;
            uint8_t blue = inPixels[i].z;
            grayPixels[i] = 0.299f*red + 0.587f*green + 0.114f*blue;
        }
    }
}

int abs(int x) {
    return x < 0 ? -1 * x : x;
}

void calcEnergy(uint8_t * grayPixels, int width, int height, int * energyMap,
                int * filterXSobel, int * filterYSobel, int filterWidth)
{
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int convolutionX = 0;
            int convolutionY = 0;

            for (int filterR = 0; filterR < filterWidth; filterR++) {
                for (int filterC = 0; filterC < filterWidth; filterC++) {
                    // Calc convolution with X-Sobel filter
                    int filterValX = filterXSobel[filterR * filterWidth + filterC];
                    int grayPixelsR = r - filterWidth / 2 + filterR;
                    int grayPixelsC = c - filterWidth / 2 + filterC;
                    grayPixelsR = min(max(0, grayPixelsR), height - 1);
                    grayPixelsC = min(max(0, grayPixelsC), width - 1);
                    uint8_t grayPixel = grayPixels[grayPixelsR * width + grayPixelsC];
                    convolutionX += filterValX * (int)grayPixel;

                    // Calc convolution with Y-Sobel filter
                    int filterValY = filterYSobel[filterR * filterWidth + filterC];
                    grayPixelsR = r - filterWidth / 2 + filterR;
                    grayPixelsC = c - filterWidth / 2 + filterC;
                    grayPixelsR = min(max(0, grayPixelsR), height - 1);
                    grayPixelsC = min(max(0, grayPixelsC), width - 1);
                    grayPixel = grayPixels[grayPixelsR * width + grayPixelsC];
                    convolutionY += filterValY * (int)grayPixel;
                }
            }

            energyMap[r * width + c] = abs(convolutionX) + abs(convolutionY);
        }
    }
}

void findMinimumSeam(int * energyMap, int width, int height,
                    int * backtrack, int * L1, int * L2)
{
    memcpy(L1, energyMap, width * sizeof(int));

    for (int r = 1; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int i = r * width + c;
            int idx;
            int energyMin = 1e9;
            
            for (int k = -1; k < 2; k++) {
                if ((c + k < 0) || (c + k == width))
                    continue;

                if (energyMin > L1[c + k]) {
                    energyMin = L1[c + k];
                    idx = k;
                }
            }

            backtrack[i] = c + idx;
            L2[c] = energyMap[i] + L1[c + idx];
        }
        memcpy(L1, L2, width * sizeof(int));
    }
}

void deleteSeam(uchar3 * inPixels, int width, int height, uchar3 * outPixels,
                int * backtrack, int * L1, int * L2)
{
    int * seamPath = (int *)malloc(height * sizeof(int));
    int energyMin = 1e9, posMin;
    for (int i = 0; i < width; i++) {
        if (energyMin > L2[i]) {
            energyMin = L2[i];
            posMin = i;
        }
    }
    for (int r = height - 1; r >= 0; r--) {
        seamPath[r] = posMin;
        // printf("\n%i\n", seamPath[r]);
        posMin = backtrack[r * width + posMin];
    }

    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int i = r * width + c;
            int _i = r * (width - 1) + c;
            if (c == seamPath[r])
                continue;
            if (c > seamPath[r])
                _i--;
            outPixels[_i] = inPixels[i];
        }
    }

    free(seamPath);
}

void seamCarving(uchar3 * inPixels, int width, int height, uchar3 * outPixels,
                int * filterXSobel, int * filterYSobel, int filterWidth,
                bool useDevice = false, dim3 blockSize = dim3(1, 1), int kernelType = 1)
{
    if (useDevice == false)
    {
        uint8_t * grayPixels = (uint8_t *)malloc(width * height * sizeof(uint8_t));
        int * energyMap = (int *)malloc(width * height * sizeof(int));
        int * backtrack = (int *)malloc(width * height * sizeof(int));
        int * L1 = (int *)malloc(width * sizeof(int));
        int * L2 = (int *)malloc(width * sizeof(int));

        uchar3 * tempPixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
        memcpy(tempPixels, inPixels, width * height * sizeof(uchar3));

        for (int i = 0; i < WIDTH_REMOVE; i++) {
            convertGrayscale(tempPixels, width - i, height, grayPixels);

            calcEnergy(grayPixels, width - i, height, energyMap, filterXSobel, filterYSobel, filterWidth);

            findMinimumSeam(energyMap, width - i, height, backtrack, L1, L2);

            deleteSeam(tempPixels, width - i, height, tempPixels, backtrack, L1, L2);
        }

        for (int r = 0; r < height; r++) {
            for (int c = 0; c < width - WIDTH_REMOVE; c++) {
                int i = r * (width - WIDTH_REMOVE) + c;
                outPixels[i] = tempPixels[i];
            }
        }

        free(grayPixels);
        free(energyMap);
        free(backtrack);
        free(L1);
        free(L2);
        free(tempPixels);
    }
    else // Use device
    {
    //     GpuTimer timer;

        
    //     timer.Stop();
    //     float time = timer.Elapsed();
    //     printf("Kernel time: %f ms\n", time);
    //     cudaDeviceSynchronize();
    //     CHECK(cudaGetLastError());

    //     // Copy result from device memory
    //     CHECK(cudaMemcpy(outPixels, d_outPixels, pixelsSize, cudaMemcpyDeviceToHost));

    //     // Free device memories
    //     CHECK(cudaFree(d_inPixels));
    //     CHECK(cudaFree(d_outPixels));
    //     if (kernelType == 1 || kernelType == 2)
    //     {
    //         CHECK(cudaFree(d_filter));
    //     }
    }
}

float computeError(uchar3 *a1, uchar3 *a2, int n)
{
    float err = 0;
    for (int i = 0; i < n; i++)
    {
        err += abs((int)a1[i].x - (int)a2[i].x);
        err += abs((int)a1[i].y - (int)a2[i].y);
        err += abs((int)a1[i].z - (int)a2[i].z);
    }
    err /= (n * 3);
    return err;
}

void printError(uchar3 *deviceResult, uchar3 *hostResult, int width, int height)
{
    float err = computeError(deviceResult, hostResult, width * height);
    printf("Error: %f\n", err);
}

char *concatStr(const char *s1, const char *s2)
{
    char *result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

void printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor);
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("CMEM: %lu bytes\n", devProv.totalConstMem);
    printf("L2 cache: %i bytes\n", devProv.l2CacheSize);
    printf("SMEM / one SM: %lu bytes\n", devProv.sharedMemPerMultiprocessor);

    printf("****************************\n");
}

int main(int argc, char **argv)
{
    if (argc != 3 && argc != 5)
    {
        printf("The number of arguments is invalid\n");
        return EXIT_FAILURE;
    }

    printDeviceInfo();

    // Read input image file
    int width, height;
    uchar3 *inPixels;
    readPnm(argv[1], width, height, inPixels);
    printf("\nImage size (width x height): %i x %i\n", width, height);

    // Set up a simple filter with blurring effect
    int filterWidth = FILTER_WIDTH;
    int *filterXSobel = (int *)malloc(filterWidth * filterWidth * sizeof(int));
    int *filterYSobel = (int *)malloc(filterWidth * filterWidth * sizeof(int));
    
    filterXSobel[0] = 1, filterXSobel[1] = 0, filterXSobel[2] = -1;
    filterXSobel[3] = 2, filterXSobel[4] = 0, filterXSobel[5] = -2;
    filterXSobel[6] = 1, filterXSobel[7] = 0, filterXSobel[8] = -1;

    filterYSobel[0] = 1, filterYSobel[1] = 2, filterYSobel[2] = 1;
    filterYSobel[3] = 0, filterYSobel[4] = 0, filterYSobel[5] = 0;
    filterYSobel[6] = -1, filterYSobel[7] = -2, filterYSobel[8] = -1;

    // Blur input image not using device
    uchar3 *outPixels = (uchar3 *)malloc((width - WIDTH_REMOVE) * height * sizeof(uchar3));
    seamCarving(inPixels, width, height, outPixels, filterXSobel, filterYSobel, filterWidth);

    // Blur input image using device, kernel 1
    // dim3 blockSize(32, 32); // Default
    // if (argc == 5)
    // {
    //     blockSize.x = atoi(argv[3]);
    //     blockSize.y = atoi(argv[4]);
    // }
    // uchar3 *outPixels1 = (uchar3 *)malloc(width * height * sizeof(uchar3));
    // blurImg(inPixels, width, height, filter, filterWidth, outPixels1, true, blockSize, 1);
    // printError(outPixels1, correctOutPixels, width, height);

    // Write results to files
    char *outFileNameBase = strtok(argv[2], "."); // Get rid of extension 
    writePnm(outPixels, width - WIDTH_REMOVE, height, concatStr(outFileNameBase, "_host.pnm"));
    // writePnm(outPixels1, width, height, concatStr(outFileNameBase, "_device1.pnm"));
    // writePnm(outPixels2, width, height, concatStr(outFileNameBase, "_device2.pnm"));
    // writePnm(outPixels3, width, height, concatStr(outFileNameBase, "_device3.pnm"));

    // Free memories
    free(inPixels);
    free(filterXSobel);
    free(filterYSobel);
    free(outPixels);
    // free(outPixels1);
}

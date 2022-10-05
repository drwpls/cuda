#include <stdio.h>
#include <stdint.h>
#include <vector_functions.h>

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

void readPnm(char *fileName,
             int &width, int &height, uchar3 *&pixels)
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

void writePnm(uchar3 *pixels, int width, int height,
              char *fileName)
{
    FILE *f = fopen(fileName, "w");
    if (f == NULL)
    {
        printf("Cannot write %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    fprintf(f, "P3\n%i\n%i\n255\n", width, height);

    for (int i = 0; i < width * height; i++)
        fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);

    fclose(f);
}

__global__ void blurImgKernel(uchar3 *inPixels, int width, int height,
                              float *filter, int filterWidth,
                              uchar3 *outPixels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    float3 _sum = make_float3(0.0f, 0.0f, 0.0f);
    for (int fr = 0; fr < filterWidth; fr++)
    {
        for (int fc = 0; fc < filterWidth; fc++)
        {
            int r = y + fr - filterWidth / 2;
            int c = x + fc - filterWidth / 2;

            if (r < 0)
                r = 0;
            if (r > height - 1)
                r = height - 1;
            if (c < 0)
                c = 0;
            if (c > width - 1)
                c = width - 1;

            int fidx = fr * filterWidth + fc;
            float filter_factor = filter[fidx];
            int iidx = r * width + c;

            _sum.x += inPixels[iidx].x * filter_factor;
            _sum.y += inPixels[iidx].y * filter_factor;
            _sum.z += inPixels[iidx].z * filter_factor;
        }
    }

    outPixels[idx] = make_uchar3(_sum.x, _sum.y, _sum.z);
}

void blurImg(uchar3 *inPixels, int width, int height, float *filter, int filterWidth,
             uchar3 *outPixels,
             bool useDevice = false, dim3 blockSize = dim3(1, 1))
{
    GpuTimer timer;
    timer.Start();
    if (useDevice == false)
    {
        // TODO
        for (int r = 0; r < height; ++r)
        {
            for (int c = 0; c < width; ++c)
            {
                float3 _sum = make_float3(0.0f, 0.0f, 0.0f);

                for (int fr = 0; fr < filterWidth; ++fr)
                {
                    for (int fc = 0; fc < filterWidth; ++fc)
                    {
                        int curRow = r + fr - filterWidth / 2;
                        int curCol = c + fc - filterWidth / 2;

                        if (curRow < 0)
                            curRow = 0;
                        if (curRow > height - 1)
                            curRow = height - 1;
                        if (curCol < 0)
                            curCol = 0;
                        if (curCol > width - 1)
                            curCol = width - 1;

                        float _filterFactor = filter[fr * filterWidth + fc];

                        _sum.x += _filterFactor * inPixels[curRow * width + curCol].x;
                        _sum.y += _filterFactor * inPixels[curRow * width + curCol].y;
                        _sum.z += _filterFactor * inPixels[curRow * width + curCol].z;
                    }
                }
                outPixels[r * width + c] = make_uchar3(_sum.x, _sum.y, _sum.z);
            }
        }
    }
    else // Use device
    {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, 0);
        printf("GPU name: %s\n", devProp.name);
        printf("GPU compute capability: %d.%d\n", devProp.major, devProp.minor);

        // TODO
        uchar3 *in_pixels, *out_pixels;
        size_t n_pixels = height * width;
        float *filter_kernel;
        size_t filter_kernel_size = filterWidth * filterWidth * sizeof(float);

        CHECK(cudaMalloc(&in_pixels, n_pixels * sizeof(uchar3)));
        CHECK(cudaMalloc(&out_pixels, n_pixels * sizeof(uchar3)));
        CHECK(cudaMalloc(&filter_kernel, filter_kernel_size));

        // TODO: Copy data to device memories
        CHECK(cudaMemcpy(in_pixels, inPixels, n_pixels * sizeof(uchar3), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(filter_kernel, filter, filter_kernel_size, cudaMemcpyHostToDevice));

        // Set up (block) and grid sizes
        dim3 gridSize((width - 1) / (blockSize.x) + 1, (height - 1) / (blockSize.y) + 1, 1);
        // Call kernel and check for errors
        blurImgKernel<<<gridSize, blockSize>>>(in_pixels, width, height, filter_kernel, filterWidth, out_pixels);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());

        // Copy calculated data back to host
        CHECK(cudaMemcpy(outPixels, out_pixels, n_pixels * sizeof(uchar3), cudaMemcpyDeviceToHost));

        // Free memory
        CHECK(cudaFree(in_pixels));
        CHECK(cudaFree(out_pixels));
        CHECK(cudaFree(filter_kernel));
    }
    timer.Stop();
    float time = timer.Elapsed();
    printf("Processing time (%s): %f ms\n",
           useDevice == true ? "use device" : "use host", time);
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

char *concatStr(const char *s1, const char *s2)
{
    char *result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

int main(int argc, char **argv)
{
    if (argc != 4 && argc != 6)
    {
        printf("The number of arguments is invalid\n");
        return EXIT_FAILURE;
    }

    // Read input image file
    int width, height;
    uchar3 *inPixels;
    readPnm(argv[1], width, height, inPixels);
    printf("Image size (width x height): %i x %i\n\n", width, height);

    // Read correct output image file
    int correctWidth, correctHeight;
    uchar3 *correctOutPixels;
    readPnm(argv[3], correctWidth, correctHeight, correctOutPixels);
    if (correctWidth != width || correctHeight != height)
    {
        printf("The shape of the correct output image is invalid\n");
        return EXIT_FAILURE;
    }

    // Set up a simple filter with blurring effect
    int filterWidth = 9;
    float *filter = (float *)malloc(filterWidth * filterWidth * sizeof(float));
    for (int filterR = 0; filterR < filterWidth; filterR++)
    {
        for (int filterC = 0; filterC < filterWidth; filterC++)
        {
            filter[filterR * filterWidth + filterC] = 1. / (filterWidth * filterWidth);
        }
    }

    // Blur input image using host
    uchar3 *hostOutPixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
    blurImg(inPixels, width, height, filter, filterWidth, hostOutPixels);

    // Compute mean absolute error between host result and correct result
    float hostErr = computeError(hostOutPixels, correctOutPixels, width * height);
    printf("Error: %f\n\n", hostErr);

    // Blur input image using device
    uchar3 *deviceOutPixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
    dim3 blockSize(32, 32); // Default
    if (argc == 6)
    {
        blockSize.x = atoi(argv[4]);
        blockSize.y = atoi(argv[5]);
    }
    blurImg(inPixels, width, height, filter, filterWidth, deviceOutPixels, true, blockSize);

    // Compute mean absolute error between device result and correct result
    float deviceErr = computeError(deviceOutPixels, correctOutPixels, width * height);
    printf("Error: %f\n\n", deviceErr);

    // Write results to files
    char *outFileNameBase = strtok(argv[2], "."); // Get rid of extension
    writePnm(hostOutPixels, width, height, concatStr(outFileNameBase, "_host.pnm"));
    writePnm(deviceOutPixels, width, height, concatStr(outFileNameBase, "_device.pnm"));

    // Free memories
    free(inPixels);
    free(correctOutPixels);
    free(hostOutPixels);
    free(deviceOutPixels);
    free(filter);
}

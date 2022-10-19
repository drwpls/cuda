#include <stdio.h>

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
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

__global__ void reduceBlksKernel1(int * in, int n, int * out)
{
	// TODO

}

__global__ void reduceBlksKernel2(int * in, int n, int * out)
{
	// TODO

}

__global__ void reduceBlksKernel3(int * in, int n, int * out)
{
	// TODO

}

int reduce(int const * in, int n,
        bool useDevice=false, dim3 blockSize=dim3(1), int kernelType=1)
{

	int result = 0; // Init
	if (useDevice == false)
	{
		result = in[0];
		for (int i = 1; i < n; i++)
		{
			result += in[i];
		}
	}
	else // Use device
	{
		// Allocate device memories
		int * d_in, * d_out;
		dim3 gridSize(1); // TODO: Compute gridSize from n and blockSize
		CHECK(cudaMalloc(&d_in, n * sizeof(int)));
		CHECK(cudaMalloc(&d_out, gridSize.x * sizeof(int)));

		// Copy data to device memory
		CHECK(cudaMemcpy(d_in, in, n * sizeof(int), cudaMemcpyHostToDevice));

		// Call kernel
		GpuTimer timer;
		timer.Start();
		if (kernelType == 1)
			reduceBlksKernel1<<<gridSize, blockSize>>>(d_in, n, d_out);
		else if (kernelType == 2)
			reduceBlksKernel2<<<gridSize, blockSize>>>(d_in, n, d_out);
		else
			reduceBlksKernel3<<<gridSize, blockSize>>>(d_in, n, d_out);
		timer.Stop();
		float kernelTime = timer.Elapsed();
		cudaDeviceSynchronize();
		CHECK(cudaGetLastError());

		// Copy result from device memory
		int * out = (int *)malloc(gridSize.x * sizeof(int));
		CHECK(cudaMemcpy(out, d_out, gridSize.x * sizeof(int), cudaMemcpyDeviceToHost));

		// Free device memories
		CHECK(cudaFree(d_in));
		CHECK(cudaFree(d_out));

		// Host do the rest of the work
		timer.Start();
		result = out[0];
		for (int i = 1; i < gridSize.x; i++)
			result += out[i];
		timer.Stop();
		float postKernelTime = timer.Elapsed();

		// Free memory
		free(out);

		// Print info
		printf("\nKernel %d\n", kernelType);
		printf("Grid size: %d, block size: %d\n", gridSize.x, blockSize.x);
		printf("Kernel time = %f ms, post-kernel time = %f ms\n", kernelTime, postKernelTime);
	}

	return result;
}

void checkCorrectness(int r1, int r2)
{
	if (r1 == r2)
		printf("CORRECT :)\n");
	else
		printf("INCORRECT :(\n");
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
    printf("****************************\n\n");

}

int main(int argc, char ** argv)
{
	printDeviceInfo();

	// Set up input size
    int n = (1 << 24) + 1;
    printf("Input size: %d\n", n);

    // Set up input data
    int * in = (int *) malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
    {
        // Generate a random integer in [0, 255]
        in[i] = (int)(rand() & 0xFF);
    }

    // Reduce NOT using device
    int correctResult = reduce(in, n);

    // Reduce using device, kernel1
    dim3 blockSize(512); // Default
    if (argc == 2)
    	blockSize.x = atoi(argv[1]);
    int result1 = reduce(in, n, true, blockSize, 1);
    checkCorrectness(result1, correctResult);

    // Reduce using device, kernel2
    int result2 = reduce(in, n, true, blockSize, 2);
    checkCorrectness(result2, correctResult);

    // Reduce using device, kernel3
    int result3 = reduce(in, n, true, blockSize, 3);
    checkCorrectness(result3, correctResult);

    // Free memories
    free(in);
}

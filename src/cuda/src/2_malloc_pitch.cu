#include <cudaDefs.h>
#include <iostream>

#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

constexpr unsigned int TPB = 8;				//8*8=64 threads per block

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

__global__ void fillData(const unsigned int pitch, const unsigned int rows, const unsigned int cols, float* data)
{
	//TODO: fill data
	const unsigned int ix = (blockIdx.x * blockDim.x) + threadIdx.x; // rows
	const unsigned int iy = (blockIdx.y * blockDim.y) + threadIdx.y; // cols

	if ((ix < rows) && (iy < cols))
	{
		// Recalculation of the index in the 2D array
		uint32_t pitchInElements = pitch / sizeof(float); // length of the row in memory, height of the (pitched) matrix column
		uint32_t index = iy * pitchInElements + ix;
		data[index] = ix + iy * rows;
	}
}


__global__ void fillDataPointer(const unsigned int pitch, const unsigned int rows, const unsigned int cols, float* data)
{
	//TODO: fill data
	const unsigned int ix = (blockIdx.x * blockDim.x) + threadIdx.x; // rows
	const unsigned int iy = (blockIdx.y * blockDim.y) + threadIdx.y; // cols

	if ((ix < rows) && (iy < cols))
	{
		// Pointer arithmetic
		float* ptr = (float*)((char*)data + iy * pitch) + ix;
		*ptr = ix + iy * rows;
	}
}


int main(int argc, char* argv[])
{
	initializeCUDA(deviceProp);

	float* devPtr;
	size_t pitch;

	constexpr unsigned int n_rows = 5;
	constexpr unsigned int n_cols = 10;

	// Allocate Pitch memory - COLUMN major
	// (columns are stacked one after another in memory)
	checkCudaErrors(cudaMallocPitch((void**)&devPtr, &pitch, n_rows * sizeof(float), n_cols));

	// Prepare grid, blocks
	dim3 dimGrid{ (n_rows + TPB - 1) / TPB, (n_cols + TPB - 1) / TPB, 1 }; // Column major
	dim3 dimBlock{ TPB, TPB, 1 }; // 2D block (TPB * TPB)

	//Call kernel
	fillData KERNEL_ARGS2(dimGrid, dimBlock)(pitch, n_rows, n_cols, devPtr);

	// Allocate Host memory and copy back Device data
	float* hostPtr = static_cast<float*>(malloc(n_rows * n_cols * sizeof(float))); // or `new float[n_rows * n_cols]`;
	checkCudaErrors(cudaMemcpy2D(
		hostPtr, // destination pointer
		n_rows * sizeof(float), // destination pitch bytes
		devPtr, // source pointer
		pitch, // width (column length for a column major matrix)
		n_rows * sizeof(float), // bytes
		n_cols, // height (number of column for a column major matrix)
		cudaMemcpyDeviceToHost));

	// Check data
	for (unsigned int i = 0; i < n_rows; i++)
	{
		for (unsigned int j = 0; j < n_cols; j++)
		{
			// Skip 'j' columns and add 'i' rows (column major)
			printf("%3.0f ", hostPtr[j * n_rows + i]);
		}
		std::cout << std::endl;
	}

	checkDeviceMatrix(devPtr, pitch, n_cols, n_rows, "%5.0f", "DEVICE: rows represent columns (column major)");

	// Free memory
	delete[] hostPtr;
	cudaFree(devPtr);

	return 0;
}

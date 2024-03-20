#include <cudaDefs.h>
#include <iostream>

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

#pragma region CustomStructure
typedef struct __align__(8) CustomStructure
{
public:
	int dim;				//dimension
	int noRecords;			//number of Records

	CustomStructure& operator=(const CustomStructure & other)
	{
		dim = other.dim;
		noRecords = other.noRecords;
		return *this;
	}

	inline void print()
	{
		printf("Dimension: %u\n", dim);
		printf("Number of Records: %u\n", noRecords);
	}
}CustomStructure;
#pragma endregion


// __constant__ __device__ variables are globally accessible by all threads
// in all blocks and are being interpreted as pointers to global memory space
__constant__ __device__ int dScalarValue;
__constant__ __device__ struct CustomStructure dCustomStructure;
__constant__ __device__ int dConstantArray[20];


__global__ void kernelConstantStruct(int* data, const unsigned int dataLength)
{
	unsigned int threadOffset = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadOffset < dataLength)
		data[threadOffset] = dCustomStructure.dim;
}

__global__ void kernelConstantArray(int* data, const unsigned int dataLength)
{
	unsigned int threadOffset = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadOffset < dataLength)
		data[threadOffset] = dConstantArray[0];
}

int main(int argc, char* argv[])
{
	initializeCUDA(deviceProp);

	// Test 0 - scalar Value
	int hScalarValue = deviceProp.multiProcessorCount;
	cudaMemcpyToSymbol(dScalarValue, static_cast<const void*>( & hScalarValue), sizeof(int));

	int hScalarValue2;
	cudaMemcpyFromSymbol(static_cast<void*>( & hScalarValue2), dScalarValue, sizeof(int));

	std::cout << hScalarValue2 << std::endl;

	// Test 1 - structure
	CustomStructure hCustomStructure = {2, 3};
	cudaMemcpyToSymbol(dCustomStructure, static_cast<const void*>(&hCustomStructure), sizeof(CustomStructure));
	
	CustomStructure hCustomStructure2;
	cudaMemcpyFromSymbol(static_cast<void*>(&hCustomStructure2), dCustomStructure, sizeof(CustomStructure));

	hCustomStructure2.print();

	// Test2 - array
	int hConstantArray[20];
	for (int i = 0; i < 20; i++)
		hConstantArray[i] = i;
	cudaMemcpyToSymbol(dConstantArray, static_cast<const void*>(hConstantArray), sizeof(int) * 20);

	int hConstantArray2[20];
	cudaMemcpyFromSymbol(hConstantArray2, dConstantArray, sizeof(int) * 20);

	for (int i = 0; i < 20; i++)
		std::cout << hConstantArray2[i] << " ";
}

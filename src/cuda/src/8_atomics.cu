#include <cudaDefs.h>
#include <limits>
#include <benchmark.h>

#define __PRINT__  cout <<  __PRETTY_FUNCTION__ <<  endl

constexpr unsigned int TPB = 512;
constexpr unsigned int NO_BLOCKS = 56;
constexpr unsigned int N = 1 << 28;

constexpr int numberOfPasses = 2; // how many times do we run the benchmark?

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

int* a, * b;
int* da, * db, * dGlobalMax;

__host__ void fillData(int* data, const int length)
{
	for (int i = 0; i < length; i++)
	{
		data[i] = i;
	}
	data[static_cast<int>(length * 0.5)] = length;
}

__host__ void fillData(int* data, const int length, const unsigned int value)
{
	for (int i = 0; i < length; i++)
	{
		data[i] = i;
	}
}

__host__ void prepareData()
{
	// Page-locked allocation
	//------------------------
	constexpr unsigned int aSize = N * sizeof(int);
	constexpr unsigned int bSize = NO_BLOCKS * sizeof(int);

	cudaHostAlloc((void**)&a, aSize, cudaHostAllocDefault);
	cudaHostAlloc((void**)&b, bSize, cudaHostAllocDefault); // helper array for aggregations within blocks

	fillData(a, N);
	fillData(b, NO_BLOCKS, INT_MIN);

	cudaMalloc((void**)&da, aSize);
	cudaMalloc((void**)&db, aSize);
	cudaMalloc((void**)&dGlobalMax, sizeof(int));

	cudaMemcpy(da, a, aSize, cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, bSize, cudaMemcpyHostToDevice);
}

__host__ void releaseData()
{
	cudaFree(da);
	cudaFree(db);
	cudaFree(dGlobalMax);

	cudaFreeHost(a);
	cudaFreeHost(da);
}

template<bool MAKE_IF>
__global__ void kernel0(const int* __restrict__ data, const unsigned int dataLength, int* __restrict__ globalMax)
{
	// This solution is limited by data length.

	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid >= dataLength) return;

	atomicMax(globalMax, data[tid]);
}

template<bool MAKE_IF>
__global__ void kernel1(const int* __restrict__ data, const unsigned int dataLength, int* __restrict__ globalMax)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	int* tData = (int*)data + tid;

	while (tid < dataLength)
	{
		if (MAKE_IF)
		{
			if((*globalMax) < (*tData))
			{
				atomicMax(globalMax, *tData);
			}
		}
		else
		{
			atomicMax(globalMax, *tData);
		}

		tData += stride;
		tid += stride;
	}
}

template<bool MAKE_IF>
__global__ void kernel2(
	const int* __restrict__ data,
	const unsigned int dataLength,
	int* __restrict__ localMax,
	const unsigned int localMaxLength,
	int* __restrict__ globalMax
){
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	int* tData = (int*)data + tid;

	while (tid < dataLength)
	{
		if (MAKE_IF)
		{
			if (localMax[blockIdx.x] < *tData)
			{
				atomicMax(&localMax[blockIdx.x], *tData);
			}
		}
		else
		{
			atomicMax(&localMax[blockIdx.x], *tData);
		}

		tData += stride;
		tid += stride;
	}

	__syncthreads();

	// 0th thread
	if (threadIdx.x == 0)
	{
		atomicMax(globalMax, localMax[blockIdx.x]);
	}
}


template<bool MAKE_IF>
__global__ void kernel3(
	const int* __restrict__ data, 
	const unsigned int dataLength,
	int* __restrict__ globalMax
){
	__shared__ int sharedMax;

	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	int* tData = (int*)data + tid;

	if(tid == 0)
	{
		sharedMax = *tData;
	}

	__syncthreads();

	while (tid < dataLength)
	{
		if (MAKE_IF)
		{
			if (sharedMax < *tData)
			{
				atomicMax(&sharedMax, *tData);
			}
		}
		else
		{
			atomicMax(&sharedMax, *tData);
		}

		tData += stride;
		tid += stride;
	}

	__syncthreads();

	if (threadIdx.x == 0)
	{
		atomicMax(globalMax, sharedMax);
	}
}

template<bool MAKE_IF>
__host__ void testKernel0()
{
	dim3 blockSize(TPB, 1, 1);
	dim3 gridSize(getNumberOfParts(N, TPB), 1, 1);

	int globalMax = INT_MIN;

	auto test = [&]() {
		cudaMemcpy(dGlobalMax, &globalMax, sizeof(int), cudaMemcpyHostToDevice);
		kernel0<MAKE_IF> << <gridSize, blockSize >> > (da, N, dGlobalMax);
		};

	float gpuTime = GPUTIME(numberOfPasses, test());
	cudaDeviceSynchronize();
	printf("\x1B[93m[GPU time] %s: %f ms\033[0m\n", __PRETTY_FUNCTION__, gpuTime);

	cudaMemcpy(&globalMax, dGlobalMax, sizeof(int), cudaMemcpyDeviceToHost);
	printf("\nMaximum: %d\n", globalMax);
}

template<bool MAKE_IF>
__host__ void testKernel1()
{
	dim3 blockSize(TPB, 1, 1);
	dim3 gridSize(getNumberOfParts(N, TPB), 1, 1);

	int globalMax = INT_MIN;

	auto test = [&]() {
		cudaMemcpy(dGlobalMax, &globalMax, sizeof(int), cudaMemcpyHostToDevice);
		kernel1<MAKE_IF> << <gridSize, blockSize >> > (da, N, dGlobalMax);
		};

	float gpuTime = GPUTIME(numberOfPasses, test());
	cudaDeviceSynchronize();
	printf("\x1B[93m[GPU time] %s: %f ms\033[0m\n", __PRETTY_FUNCTION__, gpuTime);

	cudaMemcpy(&globalMax, dGlobalMax, sizeof(int), cudaMemcpyDeviceToHost);
	printf("\nMaximum: %d\n", globalMax);
}

template<bool MAKE_IF>
__host__ void testKernel2()
{
	dim3 blockSize(TPB, 1, 1);
	dim3 gridSize(getNumberOfParts(N, TPB), 1, 1);

	int globalMax = INT_MIN;

	auto test = [&]() {
		cudaMemcpy(dGlobalMax, &globalMax, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(db, b, NO_BLOCKS * sizeof(int), cudaMemcpyHostToDevice);
		kernel2<MAKE_IF> << <gridSize, blockSize >> > (da, N, db, NO_BLOCKS, dGlobalMax);
		};

	float gpuTime = GPUTIME(numberOfPasses, test());
	cudaDeviceSynchronize();
	printf("\x1B[93m[GPU time] %s: %f ms\033[0m\n", __PRETTY_FUNCTION__, gpuTime);

	cudaMemcpy(&globalMax, dGlobalMax, sizeof(int), cudaMemcpyDeviceToHost);
	printf("\nMaximum: %d\n", globalMax);
}

template<bool MAKE_IF>
__host__ void testKernel3()
{
	dim3 blockSize(TPB, 1, 1);
	dim3 gridSize(getNumberOfParts(N, TPB), 1, 1);

	int globalMax = INT_MIN;

	auto test = [&]() {
		cudaMemcpy(dGlobalMax, &globalMax, sizeof(int), cudaMemcpyHostToDevice);
		kernel3<MAKE_IF> << <gridSize, blockSize >> > (da, N, dGlobalMax);
		};

	float gpuTime = GPUTIME(numberOfPasses, test());
	cudaDeviceSynchronize();
	printf("\x1B[93m[GPU time] %s: %f ms\033[0m\n", __PRETTY_FUNCTION__, gpuTime);

	cudaMemcpy(&globalMax, dGlobalMax, sizeof(int), cudaMemcpyDeviceToHost);
	printf("\nMaximum: %d\n", globalMax);
}

int main(int argc, char* argv[])
{
	initializeCUDA(deviceProp);
	prepareData();

	testKernel0<true>();
	//testKernel0<false>();

	testKernel1<true>();
	testKernel1<false>();

	testKernel2<true>();
	testKernel2<false>();

	testKernel3<true>();
	testKernel3<false>();

	releaseData();
	return 0;
}

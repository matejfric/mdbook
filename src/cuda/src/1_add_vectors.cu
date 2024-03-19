#include <cudaDefs.h>
#include "benchmark.h"

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

constexpr unsigned int TPB = 256; // THREADS_PER_BLOCK
constexpr unsigned int MEMBLOCKS_PER_THREADBLOCK = 2;

using namespace std;

__global__ void add1(const int* __restrict__ a, const int* __restrict__ b, const unsigned int length, int* __restrict__ c)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// # of threads in the grid
	int stride = blockDim.x * gridDim.x;

	// Grid-Stride Loop
	for (int i = tid; i < length; i += stride)
		c[i] = a[i] + b[i];
}

__global__ void add2(const int* __restrict__ a, const int* __restrict__ b, const unsigned int length, int* __restrict__ c)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// # of threads in the grid
	const unsigned int stride = blockDim.x * gridDim.x;

	while (tid < length)
	{
		c[tid] = a[tid] + b[tid];
		tid += stride;
	}
}

int main(int argc, char* argv[])
{
	initializeCUDA(deviceProp);

	constexpr unsigned int N = 1 << 10;
	constexpr unsigned int sizeInBytes = N * sizeof(int);

	// Allocate Host memory
	int* a, * b, * c;

	//x = new int[N]; // c++
	//x = (int*)malloc(sizeInBytes); // c
	//x = static_cast<int*>(::operator new(sizeInBytes)); // c++ equivalent of malloc

	a = new int[N];
	b = new int[N];
	c = new int[N];

	// Init data
	for (size_t i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = i + 1;
	}

	// Allocate Device memory
	int* da = nullptr;
	int* db = nullptr;
	int* dc = nullptr;
	checkCudaErrors(cudaMalloc((void**)&da, sizeInBytes));
	checkCudaErrors(cudaMalloc((void**)&db, sizeInBytes));
	checkCudaErrors(cudaMalloc((void**)&dc, sizeInBytes));

	// Copy Data
	checkCudaErrors(cudaMemcpy(da, a, sizeInBytes, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(db, b, sizeInBytes, cudaMemcpyHostToDevice));
	//checkDeviceMatrix<int>(da, sizeInBytes, 1, N, "%d ", "Device A");
	//checkDeviceMatrix<int>(db, sizeInBytes, 1, N, "%d ", "Device B");

	// Prepare grid and blocks
	constexpr unsigned int n_mp = 14; // number of streaming multiprocessors
	dim3 dimGrid(n_mp, 1, 1);
	dim3 dimBlock(TPB, 1, 1);
	int numBlocks = (N + TPB - 1) / TPB;

	// Call kernel
	add2 <<<dimGrid, dimBlock >>> (da, db, N, dc);

	// Check results
	checkCudaErrors(cudaMemcpy(c, dc, sizeInBytes, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	checkHostMatrix<int>(c, sizeInBytes, 1, N, "%d ", "Host C");


	// Free memory
	SAFE_DELETE_ARRAY(a); // delete[] a;
	SAFE_DELETE_ARRAY(b);
	SAFE_DELETE_ARRAY(c);

	SAFE_DELETE_CUDA(da); // cudaFree(da);
	SAFE_DELETE_CUDA(db);
	SAFE_DELETE_CUDA(dc);
}

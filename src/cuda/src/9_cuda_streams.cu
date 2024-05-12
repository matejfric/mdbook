#include <cudaDefs.h>
#include <time.h>
#include <math.h>
#include <benchmark.h>

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

constexpr unsigned int N = 1 << 20;
constexpr unsigned int MEMSIZE = N * sizeof(unsigned int);
constexpr unsigned int NO_LOOPS = 100;
constexpr unsigned int TPB = 256;
constexpr unsigned int GRID_SIZE = (N + TPB - 1) / TPB;

constexpr unsigned int NO_TEST_PHASES = 10;

void fillData(unsigned int* data, const unsigned int length)
{
	for (unsigned int i = 0; i < length; i++)
	{
		data[i] = 1;
	}
}

void printData(const unsigned int* data, const unsigned int length)
{
	if (data == 0) return;
	for (unsigned int i = 0; i < length; i++)
	{
		printf("%u ", data[i]);
	}
}


__global__ void kernel(const unsigned int* a, const unsigned int* b, const unsigned int length, unsigned int* c)
{
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = tid; i < length; i += stride)
		c[i] = a[i] + b[i];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Tests 1. - single stream, async calling </summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
void test1()
{
	unsigned int* a, * b, * c;
	unsigned int* da, * db, * dc;

	// paged-locked allocation
	checkCudaErrors(cudaHostAlloc((void**)&a, NO_LOOPS * MEMSIZE, cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc((void**)&b, NO_LOOPS * MEMSIZE, cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc((void**)&c, NO_LOOPS * MEMSIZE, cudaHostAllocDefault));

	fillData(a, NO_LOOPS * N);
	fillData(b, NO_LOOPS * N);

	// Data chunks on GPU
	checkCudaErrors(cudaMalloc((void**)&da, MEMSIZE));
	checkCudaErrors(cudaMalloc((void**)&db, MEMSIZE));
	checkCudaErrors(cudaMalloc((void**)&dc, MEMSIZE));

	// Create stream
	cudaStream_t stream1;
	checkCudaErrors(cudaStreamCreate(&stream1));
	
	auto lambda = [&]()
		{
			unsigned int dataOffset = 0;
			for (int i = 0; i < NO_LOOPS; i++)
			{
				// copy a->da, b->db
				checkCudaErrors(cudaMemcpyAsync(da, &a[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream1));
				checkCudaErrors(cudaMemcpyAsync(db, &b[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream1));

				// run the kernel in the stream
				kernel<<<GRID_SIZE, TPB, 0, stream1>>>(da, db, N, dc);

				// copy dc->c
				checkCudaErrors(cudaMemcpyAsync(&c[dataOffset], dc, MEMSIZE, cudaMemcpyDeviceToHost, stream1));

				dataOffset += N;
			}
		};
	float gpuTime = GPUTIME(NO_TEST_PHASES, lambda());

	cudaStreamSynchronize(stream1); // wait for stream to finish
	cudaStreamDestroy(stream1);
	cudaDeviceSynchronize();
	printf("\x1B[93m[GPU time] %s: %f ms\033[0m\n", __PRETTY_FUNCTION__, gpuTime);

	printData(c, 100);

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Tests 2. - two streams - depth first approach </summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
void test2()
{
	// Depth-wise data parallelism
	unsigned int* a, * b, * c;
	unsigned int* da1, * db1, * dc1;
	unsigned int* da2, * db2, * dc2;

	// paged-locked allocation
	checkCudaErrors(cudaHostAlloc((void**)&a, NO_LOOPS * MEMSIZE, cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc((void**)&b, NO_LOOPS * MEMSIZE, cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc((void**)&c, NO_LOOPS * MEMSIZE, cudaHostAllocDefault));

	fillData(a, NO_LOOPS * N);
	fillData(b, NO_LOOPS * N);

	// Data chunks on GPU
	checkCudaErrors(cudaMalloc((void**)&da1, MEMSIZE));
	checkCudaErrors(cudaMalloc((void**)&db1, MEMSIZE));
	checkCudaErrors(cudaMalloc((void**)&dc1, MEMSIZE));

	checkCudaErrors(cudaMalloc((void**)&da2, MEMSIZE));
	checkCudaErrors(cudaMalloc((void**)&db2, MEMSIZE));
	checkCudaErrors(cudaMalloc((void**)&dc2, MEMSIZE));

	// Create streams
	cudaStream_t stream1;
	cudaStream_t stream2;
	checkCudaErrors(cudaStreamCreate(&stream1));
	checkCudaErrors(cudaStreamCreate(&stream2));

	auto lambda = [&]()
		{
			unsigned int dataOffset = 0;
			for (int i = 0; i < NO_LOOPS; i+=2)
			{
				// Stream1
				//>>>>>>>>>
				checkCudaErrors(cudaMemcpyAsync(da1, &a[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream1));
				checkCudaErrors(cudaMemcpyAsync(db1, &b[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream1));

				kernel<<<GRID_SIZE, TPB, 0, stream1>>>(da1, db1, N, dc1);

				checkCudaErrors(cudaMemcpyAsync(&c[dataOffset], dc1, MEMSIZE, cudaMemcpyDeviceToHost, stream1));

				// Stream2
				//>>>>>>>>>
				dataOffset += N;

				checkCudaErrors(cudaMemcpyAsync(da2, &a[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream2));
				checkCudaErrors(cudaMemcpyAsync(db2, &b[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream2));

				kernel<<<GRID_SIZE, TPB, 0, stream2>>>(da2, db2, N, dc2);

				checkCudaErrors(cudaMemcpyAsync(&c[dataOffset], dc2, MEMSIZE, cudaMemcpyDeviceToHost, stream2));

				dataOffset += N;
			}
		};
	float gpuTime = GPUTIME(NO_TEST_PHASES, lambda());

	cudaStreamSynchronize(stream1); // wait for stream to finish
	cudaStreamSynchronize(stream2); // wait for stream to finish

	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);

	cudaDeviceSynchronize();
	printf("\x1B[93m[GPU time] %s: %f ms\033[0m\n", __PRETTY_FUNCTION__, gpuTime);

	printData(c, 100);

	cudaFree(da1);
	cudaFree(db1);
	cudaFree(dc1);

	cudaFree(da2);
	cudaFree(db2);
	cudaFree(dc2);

	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Tests 3. - two streams - breadth first approach</summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
void test3()
{
	// Breadth first data parallelism
	unsigned int* a, * b, * c;
	unsigned int* da1, * db1, * dc1;
	unsigned int* da2, * db2, * dc2;

	// paged-locked allocation
	checkCudaErrors(cudaHostAlloc((void**)&a, NO_LOOPS * MEMSIZE, cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc((void**)&b, NO_LOOPS * MEMSIZE, cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc((void**)&c, NO_LOOPS * MEMSIZE, cudaHostAllocDefault));

	fillData(a, NO_LOOPS * N);
	fillData(b, NO_LOOPS * N);

	// Data chunks on GPU
	checkCudaErrors(cudaMalloc((void**)&da1, MEMSIZE));
	checkCudaErrors(cudaMalloc((void**)&db1, MEMSIZE));
	checkCudaErrors(cudaMalloc((void**)&dc1, MEMSIZE));

	checkCudaErrors(cudaMalloc((void**)&da2, MEMSIZE));
	checkCudaErrors(cudaMalloc((void**)&db2, MEMSIZE));
	checkCudaErrors(cudaMalloc((void**)&dc2, MEMSIZE));

	// Create streams
	cudaStream_t stream1;
	cudaStream_t stream2;
	checkCudaErrors(cudaStreamCreate(&stream1));
	checkCudaErrors(cudaStreamCreate(&stream2));

	auto lambda = [&]()
		{
			unsigned int dataOffset1 = 0;
			unsigned int dataOffset2 = N;
			for (int i = 0; i < NO_LOOPS; i += 2)
			{
				// a -> da
				checkCudaErrors(cudaMemcpyAsync(da1, &a[dataOffset1], MEMSIZE, cudaMemcpyHostToDevice, stream1));
				checkCudaErrors(cudaMemcpyAsync(da2, &a[dataOffset2], MEMSIZE, cudaMemcpyHostToDevice, stream2));

				// b -> db
				checkCudaErrors(cudaMemcpyAsync(db1, &b[dataOffset1], MEMSIZE, cudaMemcpyHostToDevice, stream1));
				checkCudaErrors(cudaMemcpyAsync(db2, &b[dataOffset2], MEMSIZE, cudaMemcpyHostToDevice, stream2));

				// enqueue kernel for stream1 and stream2 (async by default)
				kernel<<<GRID_SIZE, TPB, 0, stream1>>>(da1, db1, N, dc1);
				kernel<<<GRID_SIZE, TPB, 0, stream2>>>(da2, db2, N, dc2);

				// dc -> c
				checkCudaErrors(cudaMemcpyAsync(&c[dataOffset1], dc1, MEMSIZE, cudaMemcpyDeviceToHost, stream1));
				checkCudaErrors(cudaMemcpyAsync(&c[dataOffset2], dc2, MEMSIZE, cudaMemcpyDeviceToHost, stream2));

				dataOffset1 += N;
				dataOffset2 += N;
			}
		};
	float gpuTime = GPUTIME(NO_TEST_PHASES, lambda());

	cudaStreamSynchronize(stream1); // wait for stream to finish
	cudaStreamSynchronize(stream2); // wait for stream to finish

	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);

	cudaDeviceSynchronize();
	printf("\x1B[93m[GPU time] %s: %f ms\033[0m\n", __PRETTY_FUNCTION__, gpuTime);

	printData(c, 100);

	cudaFree(da1);
	cudaFree(db1);
	cudaFree(dc1);

	cudaFree(da2);
	cudaFree(db2);
	cudaFree(dc2);

	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
}


int main(int argc, char* argv[])
{
	initializeCUDA(deviceProp);

	test1();
	test2();
	test3();

	return 0;
}

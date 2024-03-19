#include <cudaDefs.h>
#include <iostream>
#include <time.h>
#include <math.h>
#include <random>

//WARNING!!! Do not change TPB and NO_FORCES for this demo !!!
constexpr unsigned int TPB = 128;
constexpr unsigned int NO_FORCES = 256;  // number of wind turbines
constexpr unsigned int NO_RAIN_DROPS = 1 << 20; // number of rain drops

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

using namespace std;

__host__ float3* createData(const unsigned int length)
{
	//TODO: Generate float3 vectors. You can use 'make_float3' method.
	float3* data = nullptr;
	data = new float3[length];

	random_device rd;
	mt19937_64 mt(rd());
	uniform_real_distribution<float> dist(0.0f, 1.0f);

	// for (unsigned int i = 0; i < length; i++)
	// {
	// 	data[i] = make_float3(
	// 		dist(mt),
	// 		dist(mt),
	// 		dist(mt)
	// 	);
	// }

	for (unsigned int i = 0; i < length; i++)
	{
		data[i] = make_float3(1.0f, 1.0f, 1.0f);
	}

	return data;
}

__host__ void printData(const float3* data, const unsigned int length)
{
	if (data == 0) return;
	const float3* ptr = data;
	for (unsigned int i = 0; i < length; i++, ptr++)
	{
		printf("%5.2f %5.2f %5.2f ", ptr->x, ptr->y, ptr->z);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Sums the forces to get the final one using parallel reduction. 
/// 		    WARNING!!! The method was written to meet input requirements of our example, i.e. 128 threads and 256 forces  </summary>
/// <param name="dForces">	  	The forces. </param>
/// <param name="noForces">   	The number of forces. </param>
/// <param name="dFinalForce">	[in,out] If non-null, the final force. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void reduce(const float3* __restrict__ dForces, const unsigned int noForces, float3* __restrict__ dFinalForce)
{
    // WARNING!!! The method was written to meet input requirements of our example,
    // i.e. 128 threads and 256 forces

	__shared__ float3 sForces[TPB];	// static SH	//SEE THE WARNING MESSAGE !!!
	unsigned int tid = threadIdx.x;
	unsigned int next = TPB;						//SEE THE WARNING MESSAGE !!!

	//TODO: Make the reduction
	float3* src = &sForces[tid]; // pointer to the first (left) element
	float3* src2 = (float3*)&dForces[tid + next]; // pointer to the second (right) element

	*src = dForces[tid];

	src->x += src2->x;
	src->y += src2->y;
	src->z += src2->z;
	// make sure all threads have completed the first reduction step
	__syncthreads();

	next >>= 1; // 128->64 divide by 2 (shift right)

	if (tid >= next){ return; } // only half of the threads are active

	src2 = src + next;
	src->x += src2->x;
	src->y += src2->y;
	src->z += src2->z;
	__syncthreads();

	next >>= 1;

	if (tid >= next) { return; } // 64->32, threads remain - one warp - synchronous

	volatile float3* vsrc = &sForces[tid];
	volatile float3* vsrc2 = vsrc + next;

	vsrc->x += vsrc2->x;
	vsrc->y += vsrc2->y;
	vsrc->z += vsrc2->z;

	next >>= 1; // 32->16, (no need to cut down the other 16 threads - they can keep computing redundant data - save one if statement)

	vsrc2 = vsrc + next;
	vsrc->x += vsrc2->x;
	vsrc->y += vsrc2->y;
	vsrc->z += vsrc2->z;

	next >>= 1; // 16->8

	vsrc2 = vsrc + next;
	vsrc->x += vsrc2->x;
	vsrc->y += vsrc2->y;
	vsrc->z += vsrc2->z;

	next >>= 1; // 8->4

	vsrc2 = vsrc + next;
	vsrc->x += vsrc2->x;
	vsrc->y += vsrc2->y;
	vsrc->z += vsrc2->z;

	next >>= 1; // 4->2

	vsrc2 = vsrc + next;
	vsrc->x += vsrc2->x;
	vsrc->y += vsrc2->y;
	vsrc->z += vsrc2->z;

	next >>= 1; // 2->1

	vsrc2 = vsrc + next;
	vsrc->x += vsrc2->x;
	vsrc->y += vsrc2->y;
	vsrc->z += vsrc2->z;

	// zero th thread writes the result,
	// (always make sure there are no conflicts
	// when writing to global memory)
	if (tid == 0)
	{
		dFinalForce->x = vsrc->x;
		dFinalForce->y = vsrc->y;
		dFinalForce->z = vsrc->z;
	}

	// This won't work because the pointer 'vsrc' will
	// cease to exist after the kernel finishes (shared memory)!
	//dFinalForce = (float3*)&vsrc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Adds the FinalForce to every Rain drops position. </summary>
/// <param name="dFinalForce">	The final force. </param>
/// <param name="noRainDrops">	The number of rain drops. </param>
/// <param name="dRainDrops"> 	[in,out] If non-null, the rain drops positions. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void add(const float3* __restrict__ dFinalForce, const unsigned int noRainDrops, float3* __restrict__ dRainDrops)
{
	//TODO: Add the FinalForce to every Rain drops position.

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = tid; i < noRainDrops; i += stride)
	{
		float3* src = &dRainDrops[i];
		src->x += dFinalForce->x;
		src->y += dFinalForce->y;
		src->z += dFinalForce->z;
	}
}


int main(int argc, char* argv[])
{
	initializeCUDA(deviceProp);

	cudaEvent_t startEvent, stopEvent;
	float elapsedTime;

	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	cudaEventRecord(startEvent, 0);

	float3* hForces = createData(NO_FORCES);
	float3* hDrops = createData(NO_RAIN_DROPS);

	float3* dForces = nullptr;
	float3* dDrops = nullptr;
	float3* dFinalForce = nullptr;

	checkCudaErrors(cudaMalloc((void**)&dForces, NO_FORCES * sizeof(float3)));
	checkCudaErrors(cudaMemcpy(dForces, hForces, NO_FORCES * sizeof(float3), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&dDrops, NO_RAIN_DROPS * sizeof(float3)));
	checkCudaErrors(cudaMemcpy(dDrops, hDrops, NO_RAIN_DROPS * sizeof(float3), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&dFinalForce, sizeof(float3)));

	dim3 dimGrid(1, 1, 1);
	dim3 dimBlock(TPB, 1, 1);

	KernelSetting ksAdd; // Custom structure for kernel settings
	int nBlocks = (NO_RAIN_DROPS + TPB - 1) / TPB;
	const int nSMs = deviceProp.multiProcessorCount;
	int nBlocksPerSM = (nBlocks + nSMs - 1) / nSMs;
	ksAdd.dimGrid = dim3(nSMs, 1, 1);
	ksAdd.dimBlock = dim3(nBlocksPerSM, 1, 1);

	for (int i = 0; i < 1; i++)
	{
		reduce << <dimGrid, dimBlock >> > (dForces, NO_FORCES, dFinalForce);
		add << <ksAdd.dimGrid, ksAdd.dimBlock >> > (dFinalForce, NO_RAIN_DROPS, dDrops);
	}

	cudaDeviceSynchronize(); // Kernels run async

	checkDeviceMatrix<float>((float*)dFinalForce, sizeof(float3), 1, 3, "%5.2f ", "Final force");
	checkDeviceMatrix<float>((float*)dDrops, sizeof(float3), NO_RAIN_DROPS, 3, "%5.2f ", "Final Rain Drops");

	if (hForces)
		free(hForces);
	if (hDrops)
		free(hDrops);

	checkCudaErrors(cudaFree(dForces));
	checkCudaErrors(cudaFree(dDrops));

	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);

	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);

	printf("Time to get device properties: %f ms", elapsedTime);
}

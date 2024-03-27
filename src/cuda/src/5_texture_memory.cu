#include <cudaDefs.h>
#include <helper_math.h>  // normalize method
#include <imageManager.h>
#include <imageUtils.cuh>
#include <benchmark.h>

#define TPB_1D 8  // ThreadsPerBlock in one dimension
#define TPB_2D TPB_1D * TPB_1D  // ThreadsPerBlock = TPB_1D*TPB_1D (2D block)

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

using namespace gpubenchmark;
using DT = float;


__host__ TextureInfo createTextureObjectFrom2DArray(const ImageInfo<DT>& ii)
{
	TextureInfo ti;

	// Size info
	ti.size = { ii.width, ii.height, 1 };

	//Texture Data settings
	ti.texChannelDesc = cudaCreateChannelDesc<DT>();  // cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	checkCudaErrors(cudaMallocArray(
		&ti.texArrayData, &ti.texChannelDesc, ii.width, ii.height
	)); // allocate cudaArray
	checkCudaErrors(cudaMemcpyToArray(
		ti.texArrayData, 0, 0, ii.dPtr, ii.pitch * ii.height, cudaMemcpyDeviceToDevice
	)); // dPtr is already on device

	// Specify texture resource
	ti.resDesc.resType = cudaResourceTypeArray; // cudaArray
	ti.resDesc.res.array.array = ti.texArrayData; // cudaArray

	// Specify texture object parameters
	ti.texDesc.addressMode[0] = cudaAddressModeClamp; // clamp to x-border
	ti.texDesc.addressMode[1] = cudaAddressModeClamp; // clamp to y-border
	ti.texDesc.filterMode = cudaFilterModePoint; // matrix-like access
	ti.texDesc.readMode = cudaReadModeElementType; // matrix-like access
	ti.texDesc.normalizedCoords = false; // access by int coordinates [(0,h-1),(0,w-1)] or by float [(0,1),(0,1)]

	// Create texture object
	checkCudaErrors(cudaCreateTextureObject(
		&ti.texObj, &ti.resDesc, &ti.texDesc, nullptr
	)); // nullptr or change channel order (BGR/RGB)

	return ti;
}

__global__ void texKernel(
	const cudaTextureObject_t srcTex,
	const unsigned int srcWidth, 
	const unsigned int srcHeight, 
	float* dst)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < srcWidth && y < srcHeight)
	{
		dst[y * srcWidth + x] = tex2D<float>(srcTex, x, y);
	}
}

int main(int argc, char* argv[])
{
	initializeCUDA(deviceProp);
	FreeImage_Initialise(); // Initialize the FreeImage library

	// STEP 1 - load raw image data, HOST->DEVICE, with/without pitch
	ImageInfo<DT> src;
	// false - without pitch,
	prepareData<false>("c:/Users/matej/Desktop/Source/pa2/textures/terrain10x10.tif", src);

	// STEP 2 - create texture from the raw data
	TextureInfo tiSrc = createTextureObjectFrom2DArray(src);

	// STEP 3 - DO SOMETHING WITH THE TEXTURE
	dim3 block = {TPB_1D, TPB_1D, 1};
	dim3 grid{ 
		(src.width + TPB_1D - 1) / TPB_1D, 
		(src.height + TPB_1D - 1) / TPB_1D,
		1 
	};
	float* dst = nullptr;
	cudaMalloc((void**)&dst, src.width * src.height * sizeof(float));
	float gpuTime = GPUTIME(1, 
		texKernel<<<grid, block>>>(tiSrc.texObj, src.width, src.height, dst)
	);
	printf("\x1B[93m[GPU time] %s: %f ms\033[0m\n", "getBest", gpuTime);
	checkDeviceMatrix<float>(dst, src.width * sizeof(float), src.height, src.width, "%6.1f ", "dst");

	// STEP 4 - release unused data
	if (tiSrc.texObj)
		// Check that memory freed without errors.
		checkCudaErrors(cudaDestroyTextureObject(tiSrc.texObj));
	if (tiSrc.texArrayData)
		checkCudaErrors(cudaFreeArray(tiSrc.texArrayData));
	if (src.dPtr) cudaFree(src.dPtr);
	if (dst) cudaFree(dst);

	cudaDeviceSynchronize(); // Wait for the GPU launched work to complete
	error = cudaGetLastError();

	FreeImage_DeInitialise();
}

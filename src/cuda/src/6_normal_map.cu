#include <cudaDefs.h>
#include <helper_math.h> // normalize method
#include <imageManager.h>
#include <imageUtils.cuh>
#include <benchmark.h>

#define TPB_1D 8              // ThreadsPerBlock in one dimension
#define TPB_2D TPB_1D *TPB_1D // ThreadsPerBlock = TPB_1D*TPB_1D (2D block)

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

using DT = float;

__host__ TextureInfo createTextureObjectFrom2DArray(const ImageInfo<DT> &ii)
{
    TextureInfo ti;

    // Size info
    ti.size = {ii.width, ii.height, 1};

    // Texture Data settings
    ti.texChannelDesc = cudaCreateChannelDesc<DT>();

    // Allocate cudaArray and copy data into this array
    checkCudaErrors(cudaMallocArray(
        &ti.texArrayData, &ti.texChannelDesc, ii.width, ii.height));
    checkCudaErrors(cudaMemcpyToArray(
        ti.texArrayData, 0, 0, ii.dPtr, ii.pitch * ii.height, cudaMemcpyDeviceToDevice));

    // Specify texture resource
    ti.resDesc.resType = cudaResourceTypeArray;   // cudaArray
    ti.resDesc.res.array.array = ti.texArrayData; // cudaArray

    // Specify texture object parameters
    ti.texDesc.addressMode[0] = cudaAddressModeClamp; // clamp to x-border
    ti.texDesc.addressMode[1] = cudaAddressModeClamp; // clamp to y-border
    ti.texDesc.filterMode = cudaFilterModePoint;      // matrix-like access
    ti.texDesc.readMode = cudaReadModeElementType;    // matrix-like access
    ti.texDesc.normalizedCoords = false;              // access by int coordinates [(0,h-1),(0,w-1)] or by float [(0,1),(0,1)]

    // Create texture object
    checkCudaErrors(cudaCreateTextureObject(
        &ti.texObj, &ti.resDesc, &ti.texDesc, nullptr));

    return ti;
}

template <bool normalizeTexel>
__global__ void createNormalMap(
    const cudaTextureObject_t srcTex,
    const unsigned int srcWidth,
    const unsigned int srcHeight,
    const unsigned int dstPitchInElements,
    uchar3 *dst)
{
    // Integer is fine because 2D block is approx. 1024*1024+1024
    int tx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int ty = (int)(blockIdx.y * blockDim.y + threadIdx.y);

    if ((tx > srcWidth) || (ty > srcHeight))
        return;

    // 1D index for destination memory
    uint32_t dstOffset = ty * dstPitchInElements + tx;

    float tl = tex2D<float>(srcTex, tx - 1, ty - 1); // top left
    float t = tex2D<float>(srcTex, tx, ty - 1);      // top
    float tr = tex2D<float>(srcTex, tx + 1, ty - 1); // top right
    float l = tex2D<float>(srcTex, tx - 1, ty);      // left
    float r = tex2D<float>(srcTex, tx + 1, ty);      // right
    float bl = tex2D<float>(srcTex, tx - 1, ty + 1); // bottom left
    float b = tex2D<float>(srcTex, tx, ty + 1);      // bottom
    float br = tex2D<float>(srcTex, tx + 1, ty + 1); // bottom right

    // Sobel operator
    float3 normal = make_float3(
        (tr + 2.0f * r + br) - (tl + 2.0f * l + bl),
        (bl + 2.0f * b + br) - (tl + 2.0f * t + tr),
        1.0f / 2.0f);

    if (normalizeTexel)
    {
        normal = normalize(normal);
    }

    // Windows use BGR format
    uchar3 texel = make_uchar3(
        (unsigned char)((normal.z + 1) * 127.5f), // B (this reduces the range to 0-127, alternatively *255)
        (unsigned char)((normal.y + 1) * 127.5f), // G
        (unsigned char)((normal.x + 1) * 127.5f)  // R
    );

    dst[dstOffset] = texel;
}

void saveTexImage(
    const char *imageFileName,
    const uint32_t dstWidth,
    const uint32_t dstHeight,
    const uint32_t dstPitch,
    const uchar3 *dstData)
{
    FIBITMAP *tmp = FreeImage_Allocate(dstWidth, dstHeight, 24);
    unsigned int tmpPitch = FreeImage_GetPitch(tmp); // FREEIMAGE align row data ... You have to use pitch instead of width
    checkCudaErrors(cudaMemcpy2D(
        FreeImage_GetBits(tmp), tmpPitch, dstData, dstPitch, dstWidth * 3, dstHeight, cudaMemcpyDeviceToHost));
    // FreeImage_Save(FIF_BMP, tmp, imageFileName, 0);
    ImageManager::GenericWriter(tmp, imageFileName, FIF_BMP);
    FreeImage_Unload(tmp);
}

int main(int argc, char *argv[])
{
    initializeCUDA(deviceProp);
    FreeImage_Initialise();

    // STEP 1 - load raw image data, HOST->DEVICE, with/without pitch
    ImageInfo<DT> src;
    // false - without pitch,
    prepareData<false>("c:/Users/matej/Desktop/Source/pa2/textures/terrain3Kx3K.tif", src); // 8b grayscale

    // STEP 2 - create texture from the raw data
    TextureInfo tiSrc = createTextureObjectFrom2DArray(src);

    // SETP 3 - allocate pitch memory to store output image data
    size_t dst_pitch; // the device will return the pitch of the allocated memory (therefore &dst_pitch)
    uchar3 *dst = nullptr;
    checkCudaErrors(cudaMallocPitch(
        (void **)&dst, &dst_pitch, src.width * sizeof(uchar3), src.height));

    // STEP 4 - create normal map
    dim3 block = {TPB_1D, TPB_1D, 1};
    dim3 grid{
        (src.width + TPB_1D - 1) / TPB_1D,
        (src.height + TPB_1D - 1) / TPB_1D,
        1};
    float gpuTime = GPUTIME(1,
                            createNormalMap<true><<<grid, block>>>(
                                tiSrc.texObj, src.width, src.height, dst_pitch / sizeof(uchar3), dst));
    printf("\x1B[93m[GPU time] %s: %f ms\033[0m\n", "getBest", gpuTime);

    // STEP 5 - save the normal map
    saveTexImage("normalMap.bmp", src.width, src.height, dst_pitch, dst);

    // SETP 6 - release unused data
    if (tiSrc.texObj)
        checkCudaErrors(cudaDestroyTextureObject(tiSrc.texObj));
    if (tiSrc.texArrayData)
        checkCudaErrors(cudaFreeArray(tiSrc.texArrayData));
    if (dst)
        checkCudaErrors(cudaFree(dst));
    if (src.dPtr)
        checkCudaErrors(cudaFree(src.dPtr));

    cudaDeviceSynchronize();
    error = cudaGetLastError();

    FreeImage_DeInitialise();
}

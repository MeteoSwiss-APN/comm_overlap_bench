#include "Kernel.h"

__global__
void cukernel(Real* in, Real* out, const int kSize, const int iStride, const int jStride, const int kStride)
{

    int ipos = blockIdx.x * 32 + threadIdx.x;
    int jpos = blockIdx.y * 8 + threadIdx.y;

    for(int k=0; k < kSize; ++k)
    {
        out[ipos*iStride + jpos*jStride + k*kStride] = in[ipos*iStride + jpos*jStride + k*kStride];
    }

}

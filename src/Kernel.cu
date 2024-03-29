#include "Kernel.h"

__global__
void cukernel(Real __restrict__ * in, Real __restrict__* out, const int kSize, const int iStride, const int jStride, const int kStride)
{

    int ipos = blockIdx.x * 32 + threadIdx.x;
    int jpos = blockIdx.y * 8 + threadIdx.y;

#ifndef _NOPOW
    out[ipos * iStride + jpos * jStride] =
        (pow(in[ipos * iStride + jpos * jStride] * in[ipos * iStride + jpos * jStride], 3.5) +
            pow(in[ipos * iStride + jpos * jStride + kStride], 2.3));
#else
    out[ipos * iStride + jpos * jStride] =
        in[ipos * iStride + jpos * jStride] * in[ipos * iStride + jpos * jStride] * 3.5 +
        in[ipos * iStride + jpos * jStride + kStride] * 2.3;
#endif

    for (int k = 1; k < kSize - 1; ++k) {
#ifndef _NOPOW
        out[ipos * iStride + jpos * jStride + k * kStride] =
            (pow(in[ipos * iStride + jpos * jStride + k * kStride] * in[ipos * iStride + jpos * jStride + k * kStride],
                 3.5) +
                pow(in[ipos * iStride + jpos * jStride + (k + 1) * kStride], 2.3) -
                pow(in[ipos * iStride + jpos * jStride + (k - 1) * kStride], 1.3))
#else
        out[ipos * iStride + jpos * jStride + k * kStride] =
            (in[ipos * iStride + jpos * jStride + k * kStride] * in[ipos * iStride + jpos * jStride + k * kStride] *
                    3.5 +
                in[ipos * iStride + jpos * jStride + (k + 1) * kStride] * 2.3 -
                in[ipos * iStride + jpos * jStride + (k - 1) * kStride] * 1.3)
#endif
            + out[(ipos + 1) * iStride + jpos * jStride + k * kStride] +
            out[(ipos - 1) * iStride + jpos * jStride + k * kStride] +
            out[ipos * iStride + (jpos + 1) * jStride + k * kStride] +
            out[ipos * iStride + (jpos - 1) * jStride + k * kStride];
    }

#ifndef _NOPOW
    out[ipos * iStride + jpos * jStride + (kSize - 1) * kStride] =
        (pow(in[ipos * iStride + jpos * jStride + (kSize - 1) * kStride] *
                 in[ipos * iStride + jpos * jStride + (kSize - 1) * kStride],
             3.5) -
            pow(in[ipos * iStride + jpos * jStride + (kSize - 2) * kStride], 1.3));
#else
    out[ipos * iStride + jpos * jStride + (kSize - 1) * kStride] =
        (in[ipos * iStride + jpos * jStride + (kSize - 1) * kStride] *
                in[ipos * iStride + jpos * jStride + (kSize - 1) * kStride] * 3.5 -
            in[ipos * iStride + jpos * jStride + (kSize - 2) * kStride]) *
        1.3;
#endif
}

void launch_kernel(IJKSize domain, Real __restrict__* in, Real __restrict__* out, cudaStream_t& stream)
{
    dim3 threads, blocks;
    threads.x = 32;
    threads.y = 8;
    threads.z = 1;

    blocks.x = domain.isize / 32;
    blocks.y = domain.jsize / 8;
    blocks.z = 1;
    if (domain.isize % 32 != 0 || domain.jsize % 8 != 0)
        std::cout << "ERROR: Domain sizes should be multiple of 32x8" << std::endl;

    const int iStride = 1;
    const int jStride = domain.isize + cNumBoundaryLines * 2;
    const int kStride = (domain.jsize + cNumBoundaryLines * 2) * jStride;
    // clang-format off
    cukernel<<<blocks, threads, 0, stream>>>(in, out, domain.ksize, iStride, jStride, kStride);
    // clang-format on
}

#pragma once

#include "Definitions.h"
#include "IJKSize.h"
#include <iostream>
#include <cuda.h>

__global__
void cukernel(Real* in, Real* out, const int, const int, const int);

void launch_kernel(IJKSize domain, Real* in, Real* out)
{
    dim3 threads, blocks;
    threads.x = 32;
    threads.y = 8;
    threads.z = 1;

    blocks.x = domain.iSize() / 32;
    blocks.y = domain.jSize() / 8;
    blocks.z = 1;
    if(domain.iSize() % 32 != 0 || domain.jSize() % 8 != 0)
        std::cout << "ERROR: Domain sizes should be multiple of 32x8" << std::endl;

    const int iStride = 1;
    const int jStride = domain.iSize()+cNumBoundaryLines*2;
    const int kStride = (domain.jSize()+cNumBoundaryLines*2)* jStride;
    cukernel<<<blocks, threads>>>(in, out, iStride, jStride, kStride);
}



#pragma once

#include "Definitions.h"
#include "IJKSize.h"
#include <cuda.h>
#include <iostream>

__global__ void cukernel(Real* in, Real* out, const int, const int, const int);

void launch_kernel(IJKSize domain, Real* in, Real* out, cudaStream_t& stream);

#pragma once

#include "Definitions.h"
#include "IJKSize.h"
#include <iostream>
#include <cuda.h>

__global__
void cukernel(Real* in, Real* out, const int, const int, const int);

void launch_kernel(IJKSize domain, Real* in, Real* out, cudaStream_t& stream);



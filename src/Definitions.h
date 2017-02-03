#pragma once

#include <mpi.h>

//TODO mark these in namespace
const int cNumBoundaryLines = 3;
const int cCacheFlusherSize = 1024*1024*21;
const int cNumBenchmarkRepetitions = 1000;
#define N_HORIDIFF_VARS 4
#define PI ((Real)3.14159265358979323846) // pi
#ifdef SINGLEPRECISION
    typedef float Real;
    #define MPITYPE MPI_FLOAT
#else
    typedef double Real;
    #define MPITYPE MPI_DOUBLE
#endif

// macro defining empty copy constructors and assignment operators
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
    TypeName(const TypeName&);               \
    TypeName& operator=(const TypeName&)

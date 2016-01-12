#pragma once

//TODO mark these in namespace
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

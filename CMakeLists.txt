# Top Level CMakeLists.txt File
cmake_minimum_required( VERSION 3.4 )

# set the installation prefix
# Note: setting this *before* the project imposes a default that the user is able to change.
SET(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install/" CACHE PATH "installation prefix")

project( comm_overlap_benchmark )
enable_language( CXX )
set (CMAKE_CXX_STANDARD 11)

# Timers
set( ENABLE_TIMER "OFF" CACHE BOOL "Enable timer (requires boost)" )
if (ENABLE_TIMER)
    add_definitions( -DENABLE_TIMER )
endif()

# BOOST
if (ENABLE_TIMER)
    find_package(Boost 1.43.0 COMPONENTS system timer REQUIRED)
    if (Boost_FOUND)
        include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
    endif()
endif()

# MPI timers
set( ENABLE_MPI_TIMER "OFF" CACHE BOOL "Enable timer (MPI_Wtime)" )
if (ENABLE_MPI_TIMER)
    add_definitions( -DENABLE_MPI_TIMER )
endif()

# SCOREP region timers
set( ENABLE_SCOREP_TIMER "OFF" CACHE BOOL "Enable scorep region timer" )
if (ENABLE_SCOREP_TIMER)
    add_definitions( -DSCOREP_USER_ENABLE )
endif()

find_package(Threads)
set ( CMAKE_EXE_LINKER_FLAGS ${MPI_LINK_FLAGS} ${CMAKE_LINK_FLAGS} ${CMAKE_THREAD_LIBS_INIT})

# CUDA
set( CUDA_BACKEND "ON" CACHE BOOL "Compile CUDA version" )
if( CUDA_BACKEND )
    find_package( CUDA 5.0 )
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "" FORCE)
    add_definitions( -DCUDA_BACKEND)
    add_definitions( -DCUDASETDEVICE)
    set(CUDA_COMPUTE_CAPABILITY "sm_35" CACHE STRING "Compute capability for CUDA" )
    set(NVCC_FLAGS "-arch=${CUDA_COMPUTE_CAPABILITY}" ) # -Xptxas -abi=no
    list(APPEND NVCC_FLAGS "-std=c++11")
endif()

# Single precision
set( SINGLEPRECISION "OFF" CACHE BOOL "Single precision" )
if( SINGLEPRECISION )
  add_definitions( -DSINGLEPRECISION )
endif()

set( DISABLE_POW_CUDA "ON" CACHE BOOL "Disable POW in CUDA Kernel" )
if (DISABLE_POW_CUDA)
  add_definitions( -D_NOPOW )
endif()

find_package(MPI REQUIRED)
include_directories( ${MPI_INCLUDE_PATH} )

if(MPI_COMPILE_FLAGS)
    set(CMAKE_COMPILE_FLAGS ${MPI_COMPILE_FLAGS} ${CMAKE_COMPILE_FLAGS})
endif()

if(MPI_LINK_FLAGS)
    set(CMAKE_LINK_FLAGS ${MPI_LINK_FLAGS} ${CMAKE_LINK_FLAGS})
endif()

#### FORCE MPI Libraries to front by appending to CMAKE_EXE_LINKER_FLAGS
if(MPI_CXX_LIBRARIES)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS}")
    foreach(ITEM ${MPI_CXX_LIBRARIES})
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${ITEM}")
    endforeach()
endif()

message(STATUS "MPI_LINK_FLAGS: ${MPI_LINK_FLAGS}")
message(STATUS "MPI_COMPILE_FLAGS: ${MPI_COMPILE_FLAGS}")
message(STATUS "MPI_LIBRARIES: ${MPI_LIBRARIES}")

set( MPI_VENDOR "unknown" CACHE STRING "mpi vendor" )
if( MPI_VENDOR STREQUAL "mvapich2")
    add_definitions ( -DMVAPICH2 )
elseif( MPI_VENDOR STREQUAL "openmpi" )
    add_definitions ( -DOPENMPI )
endif()

set (VERBOSE "OFF" CACHE BOOL "VERBOSE")
if(VERBOSE)
    add_definitions(-DVERBOSE)
endif()


if ( SINGLEPRECISION )
    set ( STELLA_PRECISION "float" )
else ( SINGLEPRECISION )
    set ( STELLA_PRECISION "double" )
endif ( SINGLEPRECISION )

# CUDA streams
set ( ENABLE_CUDA_STREAMS "ON" CACHE BOOL "Enable usage of CUDA streams" )
if ( ENABLE_CUDA_STREAMS )
    add_definitions ( -DENABLE_CUDA_STREAMS )
endif( ENABLE_CUDA_STREAMS )

# Produce position independent code on Linux
if( CMAKE_HOST_UNIX AND SHARED_LIBRARY )
    message(STATUS "Compiling shared library on Unix")
    set( CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -fPIC )
endif()

include(clang-format.cmake)
# 2. build sources
add_subdirectory( src )


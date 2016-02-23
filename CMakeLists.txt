# Top Level CMakeLists.txt File
cmake_minimum_required( VERSION 2.8.11.1 )

# set the installation prefix
# Note: setting this *before* the project imposes a default that the user is able to change.
SET(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install/" CACHE PATH "installation prefix")

project( hori_diff_tests )
enable_language( CXX )

# BOOST
find_package(Boost 1.43.0 COMPONENTS system timer REQUIRED)
if (Boost_FOUND)
    include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
endif()

# OpenMP
set( ENABLE_OPENMP "OFF" CACHE BOOL "Disable OpenMP threading" )
if( ENABLE_OPENMP )
    find_package(OpenMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_definitions("-D__ENABLE_OPENMP__")
    set ( CMAKE_EXE_LINKER_FLAGS "${CMAKE_LINK_FLAGS} ${OpenMP_CXX_FLAGS}" )
else()
    ## if not compiled and link with -fopenmp, gtest will depend on pthread lib
    find_package(Threads)
    set ( CMAKE_EXE_LINKER_FLAGS "${CMAKE_LINK_FLAGS} ${CMAKE_THREAD_LIBS_INIT}")
endif( ENABLE_OPENMP )

# CUDA
set( CUDA_BACKEND "ON" CACHE BOOL "Compile CUDA version" )
if( CUDA_BACKEND )
    find_package( CUDA 5.0 )
    set(CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "" FORCE)
    add_definitions( -D__CUDA_BACKEND__ )
    set(HORIDIFF_CUDA_COMPUTE_CAPABILITY "sm_35" CACHE STRING "Compute capability for CUDA" )
    set(HORIDIFF_NVCC_FLAGS "-arch=${HORIDIFF_CUDA_COMPUTE_CAPABILITY}" ) # -Xptxas -abi=no
    set( ENABLE_OPENMP OFF CACHE BOOL "" FORCE)
else()
    message(FATAL_ERROR "Only cuda compilation is supported")
endif()

# Single precision
set( SINGLEPRECISION "OFF" CACHE BOOL "Single precision" )
if( SINGLEPRECISION )
  add_definitions( -DSINGLEPRECISION )
endif()

set( MPI_VENDOR "mvapich2" CACHE STRING "mpi vendor" )
if( MPI_VENDOR STREQUAL "mvapich2")
    add_definitions ( -DMVAPICH2 )
elseif( NOT MPI_VENDOR STREQUAL "openmpi" )
    message([FATA_ERROR] "mpi vendor not known")
endif()

set (VERBOSE "OFF" CACHE BOOL "VERBOSE")
if(VERBOSE)
    add_definitions(-DVERBOSE)
endif()

# Search for INSTALL_DIR env variable
if( NOT EXISTS "$ENV{INSTALL_DIR}" )
    set (STELLA_INSTALL_ENV "/" )
else( NOT EXISTS "$ENV{INSTALL_DIR}" )
    string ( TOLOWER "${CMAKE_BUILD_TYPE}" STELLA_BUILD_TYPE )
    if (STELLA_BUILD_TYPE STREQUAL "")
        set (STELLA_BUILD_TYPE "debug")
    endif (STELLA_BUILD_TYPE STREQUAL "")

    if ( SINGLEPRECISION )
        set ( STELLA_PRECISION "float" )
    else ( SINGLEPRECISION )
        set ( STELLA_PRECISION "double" )
    endif ( SINGLEPRECISION )

    set (STELLA_INSTALL_ENV "$ENV{INSTALL_DIR}/stella/trunk/${STELLA_BUILD_TYPE}_${STELLA_PRECISION}" )
    message (STATUS "Searching for STELLA in ${STELLA_INSTALL_ENV}")
endif( NOT EXISTS "$ENV{INSTALL_DIR}" )

# STELLA
set (STELLA_DIR "STELLA_DIR-NOTFOUND" CACHE PATH "The directory containing STELLA")

set(STELLA_INCLUDE_PATH "STELLA_SharedInfrastructure-NOTFOUND")
mark_as_advanced(STELLA_INCLUDE_PATH)
find_path(STELLA_INCLUDE_PATH
    "StencilFramework.h"
    PATHS "${STELLA_DIR}/install/include/STELLA"
          "${STELLA_DIR}/include/STELLA"
          "${STELLA_INSTALL_ENV}/include/STELLA"
          "${CMAKE_CURRENT_SOURCE_DIR}/../STELLA/install/include/STELLA" 
)
if ( NOT STELLA_INCLUDE_PATH )
    message(SEND_ERROR
            "Error: STELLA include directory has not been found.\n"
            "Please specify the variable STELLA_DIR"
    )
else ( NOT STELLA_INCLUDE_PATH )
    include_directories( SYSTEM "${STELLA_INCLUDE_PATH}" )
    message (STATUS "STELLA headers found at ${STELLA_INCLUDE_PATH}")
endif ( NOT STELLA_INCLUDE_PATH )

if (CUDA_BACKEND)
    set (STELLA_SUFFIX "CUDA")
else (CUDA_BACKEND)
    set (STELLA_SUFFIX "")
endif (CUDA_BACKEND)

set(STELLA_SharedInfrastructure "STELLA_SharedInfrastructure-NOTFOUND")
find_library(STELLA_SharedInfrastructure "SharedInfrastructure${STELLA_SUFFIX}"
    PATHS "${STELLA_DIR}/install/lib"
          "${STELLA_DIR}/lib"
          "${STELLA_INSTALL_ENV}/lib"
          "${CMAKE_CURRENT_SOURCE_DIR}/../STELLA/install/lib" 
)
mark_as_advanced(STELLA_SharedInfrastructure)
if ( NOT STELLA_SharedInfrastructure )
    message(SEND_ERROR
            "Error: SharedInfrastructure has not been found.\n"
            "Please specify the variable STELLA_DIR"
    )
else ()
    add_library(SharedInfrastructure SHARED IMPORTED)
    set_target_properties(SharedInfrastructure PROPERTIES IMPORTED_LOCATION "${STELLA_SharedInfrastructure}")
    message (STATUS "SharedInfrastructure has been found in ${STELLA_SharedInfrastructure}")
endif ()

set(STELLA_Stella "STELLA_Stella-NOTFOUND")
find_library(STELLA_Stella "Stella${STELLA_SUFFIX}"
    PATHS "${STELLA_DIR}/install/lib"
          "${STELLA_DIR}/lib"
          "${STELLA_INSTALL_ENV}/lib"
          "${CMAKE_CURRENT_SOURCE_DIR}/../STELLA/install/lib" 
)
mark_as_advanced(STELLA_Stella)
if ( NOT STELLA_Stella )
    message (SEND_ERROR
        "Error: Stella STELLA_Stella has not been found.\n"
        "Please specify the variable STELLA_DIR"
    )
else ()
    add_library(Stella SHARED IMPORTED)
    set_target_properties(Stella PROPERTIES IMPORTED_LOCATION "${STELLA_Stella}")
    message (STATUS "Stella has been found in ${STELLA_Stella}")
endif()

set(STELLA_SerializationFramework "STELLA_SerializationFramework-NOTFOUND")
find_library(STELLA_SerializationFramework "SerializationFramework${STELLA_SUFFIX}"
    PATHS "${STELLA_DIR}/install/lib"
          "${STELLA_DIR}/lib"
          "${STELLA_INSTALL_ENV}/lib"
          "${CMAKE_CURRENT_SOURCE_DIR}/../STELLA/install/lib" 
)
mark_as_advanced(STELLA_SerializationFramework)
if ( NOT STELLA_SerializationFramework)
    message (SEND_ERROR
        "Error: SerializationFramework has not been found.\n"
        "Please specify the variable STELLA_DIR"
    )
else ( NOT STELLA_SerializationFramework)
    add_library(SerializationFramework SHARED IMPORTED)
    set_target_properties(SerializationFramework PROPERTIES IMPORTED_LOCATION "${STELLA_SerializationFramework}")
    message (STATUS "SerializationFramework has been found in ${STELLA_SerializationFramework}")
endif ( NOT STELLA_SerializationFramework)
add_definitions(-D__ENABLE_SERIALIZATION__)

set(STELLA_CommunicationFramework "STELLA_CommunicationFramework-NOTFOUND")
find_library(STELLA_CommunicationFramework "CommunicationFramework${STELLA_SUFFIX}"
    PATHS "${STELLA_DIR}/install/lib"
          "${STELLA_DIR}/lib"
          "${STELLA_INSTALL_ENV}/lib"
          "${CMAKE_CURRENT_SOURCE_DIR}/../STELLA/install/lib" 
)
mark_as_advanced(STELLA_CommunicationFramework)
if ( NOT STELLA_CommunicationFramework)
    message (SEND_ERROR
        "Error: CommunicationFramework has not been found.\n"
        "Please specify the variable STELLA_DIR"
    )
else ( NOT STELLA_CommunicationFramework)
    add_library(CommunicationFramework SHARED IMPORTED)
    set_target_properties(CommunicationFramework PROPERTIES IMPORTED_LOCATION "${STELLA_CommunicationFramework}")
    message (STATUS "CommunicationFramework has been found in ${STELLA_CommunicationFramework}")
endif ( NOT STELLA_CommunicationFramework)

set(STELLA_VerificationFramework "STELLA_VerificationFramework-NOTFOUND")
find_library(STELLA_VerificationFramework "VerificationFramework${STELLA_SUFFIX}"
    PATHS "${STELLA_DIR}/install/lib"
          "${STELLA_DIR}/lib"
          "${STELLA_INSTALL_ENV}/lib"
          "${CMAKE_CURRENT_SOURCE_DIR}/../STELLA/install/lib" 
)
mark_as_advanced(STELLA_VerificationFramework)
if ( NOT STELLA_VerificationFramework)
    message (SEND_ERROR
        "Error: VerificationFramework has not been found.\n"
        "Please specify the variable STELLA_DIR"
    )
else ( NOT STELLA_VerificationFramework)
    add_library(VerificationFramework SHARED IMPORTED)
    set_target_properties(VerificationFramework PROPERTIES IMPORTED_LOCATION "${STELLA_VerificationFramework}")
    message (STATUS "VerificationFramework has been found in ${STELLA_VerificationFramework}")
endif ( NOT STELLA_VerificationFramework)

set(STELLA_StellaUtils "STELLA_StellaUtils-NOTFOUND")
find_library(STELLA_StellaUtils "StellaUtils"
    PATHS "${STELLA_DIR}/install/lib"
          "${STELLA_DIR}/lib"
          "${STELLA_INSTALL_ENV}/lib"
          "${CMAKE_CURRENT_SOURCE_DIR}/../STELLA/install/lib" 
)
mark_as_advanced(STELLA_StellaUtils)
if ( NOT STELLA_StellaUtils)
    message (SEND_ERROR
        "Error: StellaUtils has not been found.\n"
        "Please specify the variable STELLA_DIR"
    )
else ( NOT STELLA_StellaUtils)
    add_library(StellaUtils SHARED IMPORTED)
    set_target_properties(StellaUtils PROPERTIES IMPORTED_LOCATION "${STELLA_StellaUtils}")
    message (STATUS "StellaUtils has been found in ${STELLA_StellaUtils}")
endif ( NOT STELLA_StellaUtils)

set(LIBJSON "LIBJSON-NOTFOUND")
find_library(LIBJSON "json"
    PATHS "${STELLA_DIR}/install/lib"
          "${STELLA_DIR}/lib"
          "${STELLA_INSTALL_ENV}/lib"
          "${CMAKE_CURRENT_SOURCE_DIR}/../STELLA/install/lib" 
)
mark_as_advanced(LIBJSON)
add_library(libjson SHARED IMPORTED)
set_target_properties(libjson PROPERTIES IMPORTED_LOCATION "${LIBJSON}")


set(STELLA_gtest "GTEST-NOTFOUND")
find_library(STELLA_gtest "gmock-gtest"
    PATHS "${STELLA_DIR}/install/lib"
          "${STELLA_DIR}/lib"
          "${STELLA_INSTALL_ENV}/lib"
          "${CMAKE_CURRENT_SOURCE_DIR}/../STELLA/install/lib" 
)
mark_as_advanced(STELLA_gtest)

if ( NOT STELLA_gtest )
    message (SEND_ERROR
        "Error: Stella STELLA_gtest has not been found.\n"
        "Please specify the variable STELLA_DIR"
    )
else ()
    add_library(StellaGtest SHARED IMPORTED)
    set_target_properties(StellaGtest PROPERTIES IMPORTED_LOCATION "${STELLA_gtest}")
    message (STATUS "gtest has been found in ${STELLA_gtest}")
endif()

add_library(gtest SHARED IMPORTED)
set_target_properties(gtest PROPERTIES IMPORTED_LOCATION "${GTEST}")

set(SHA256 "SHA256-NOTFOUND")
find_library(SHA256 "sha256"
    PATHS "${STELLA_DIR}/install/lib"
          "${STELLA_DIR}/lib"
          "${STELLA_INSTALL_ENV}/lib"
          "${CMAKE_CURRENT_SOURCE_DIR}/../STELLA/install/lib" 
)
mark_as_advanced(SHA256)
add_library(sha256 SHARED IMPORTED)
set_target_properties(sha256 PROPERTIES IMPORTED_LOCATION "${SHA256}")

# STELLA libraries
# Here we list the STELLA libraries we have to compile against
set (STELLA_LIBRARIES SharedInfrastructure CommunicationFramework VerificationFramework SerializationFramework StellaUtils libjson sha256 StellaGtest)

# GCL
set( GCL "OFF" CACHE BOOL "Compile GCL version" )
if( GCL )
    # MPI
    find_package(MPI REQUIRED)
    include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
    set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS}" )

    # Include
    set(GCL_INCLUDE_PATH "GCL_INCLUDE_PATH-NOTFOUND")
    mark_as_advanced(GCL_INCLUDE_PATH)
    find_path(GCL_INCLUDE_PATH
        "L3/include/GCL.h"
        PATHS "${STELLA_INCLUDE_PATH}/gcl"
              "${STELLA_DIR}/install/include/STELLA/gcl"
              "${STELLA_DIR}/include/STELLA/gcl"
              "${STELLA_INSTALL_ENV}/include/STELLA/gcl"
              "${CMAKE_CURRENT_SOURCE_DIR}/../STELLA/install/include/STELLA/gcl" 
    )
    if ( NOT GCL_INCLUDE_PATH )
        message(SEND_ERROR
                "Error: GCL headers have not been found.\n"
                "Please make sure STELLA was compiled with GCL support"
        )
    else ( NOT GCL_INCLUDE_PATH )
        include_directories(SYSTEM "${GCL_INCLUDE_PATH}" )
        include_directories(SYSTEM "${GCL_INCLUDE_PATH}/L3/include" )
        include_directories(SYSTEM "${GCL_INCLUDE_PATH}/L2/include" )
        include_directories(SYSTEM "${GCL_INCLUDE_PATH}/utils" )
        message (STATUS "GCL headers found at ${GCL_INCLUDE_PATH}")
    endif ( NOT GCL_INCLUDE_PATH )

    # Search for library
    set(GCL_LIBRARY "GCL_LIBRARY-NOTFOUND")
    mark_as_advanced(GCL_LIBRARY)
    find_library(GCL_LIBRARY "gcl"
        PATHS "${STELLA_DIR}/install/lib"
              "${STELLA_DIR}/lib"
              "${STELLA_INSTALL_ENV}/lib"
              "${CMAKE_CURRENT_SOURCE_DIR}/../STELLA/install/lib" 
    )
    if ( NOT GCL_LIBRARY )
        message (SEND_ERROR
            "Error: GCL has not been found.\n"
            "Please specify the variable STELLA_DIR"
        )
    else ( NOT GCL_LIBRARY )
        add_library(gcl SHARED IMPORTED)
        set_target_properties(gcl PROPERTIES IMPORTED_LOCATION "${GCL_LIBRARY}")
        message (STATUS "GCL has been found in ${GCL_LIBRARY}")
    endif ( NOT GCL_LIBRARY )

    # Definitions
    add_definitions( -D__GCL__ )
    add_definitions( -D_GCL_MPI_ )
    if( CUDA_BACKEND )
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GCL_GPU_")
        include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
    endif( CUDA_BACKEND )

    # Update STELLA_LIBRARIES
    set(STELLA_LIBRARIES ${STELLA_LIBRARIES} "gcl" ${MPI_CXX_LIBRARIES})
endif( GCL )

# Performance meters
set ( ENABLE_PERFORMANCE_METERS "OFF" CACHE BOOL "Enable availability of meters for measurement of performance" )
if ( ENABLE_PERFORMANCE_METERS )
    add_definitions ( -DENABLE_PERFORMANCE_METERS )
    message(WARNING "Performance Meters synchronize Host and GPU streams, therefore they affect overlap of comm/comp. We recommend to deactivate these meters, unless you know what you are doing.")
endif( ENABLE_PERFORMANCE_METERS )

# CUDA streams
set ( ENABLE_CUDA_STREAMS "ON" CACHE BOOL "Enable usage of CUDA streams" )
if ( ENABLE_CUDA_STREAMS )
    add_definitions ( -DENABLE_CUDA_STREAMS )
endif( ENABLE_CUDA_STREAMS )

# Produce position independent code on Linux
if( CMAKE_HOST_UNIX AND SHARED_LIBRARY )
    message(STATUS "Compiling shared library on Unix")
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC" )
endif()

# 2. build sources
add_subdirectory( src )


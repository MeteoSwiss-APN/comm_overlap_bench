set( 
    SOURCES
    "main.cpp"
    "Definitions.h"
    "HorizontalDiffusionSA.h"
    "IJKSize.h"
    "Kernel.h"
    "MPIHelper.h"
    "MPIHelper.cpp"
    "Repository.h"
    "Repository.cpp"
    "SimpleStorage.h"
    "SimpleSwappableStorage.h"
    "Options.h"
    "Options.cpp"
) 

set(
    X86_SOURCES
    "HorizontalDiffusionSA.cpp"
)

set(
    CUDA_SOURCES
    "HorizontalDiffusionSA.cu"
    "Kernel.cu"
)

add_executable(comm_overlap_benchmark
    ${SOURCES})

#cuda and non cuda compilation
if( CUDA_BACKEND )
    cuda_add_library(
        # StandaloneStencils STATIC
        StandaloneStencils SHARED
        ${CUDA_SOURCES} 
        OPTIONS ${NVCC_FLAGS}
    )
else()
    add_library(
        StandaloneStencils STATIC 
        ${X86_SOURCES}
    )
endif()

if (${CMAKE_COMPILE_FLAGS})
    set_target_properties(comm_overlap_benchmark PROPERTIES COMPILE_FLAGS ${CMAKE_COMPILE_FLAGS})
endif()

if (${CMAKE_LINK_FLAGS})
    set_target_properties(comm_overlap_benchmark PROPERTIES LINK_FLAGS ${CMAKE_LINK_FLAGS}) 
endif()

target_link_libraries(comm_overlap_benchmark
    StandaloneStencils
    ${MPI_LIBRARIES}
)

cuda_add_executable(comm_overlap_benchmark_cc
    ${SOURCES}
    ${CUDA_SOURCES}
    OPTIONS ${NVCC_FLAGS})

if (${CMAKE_COMPILE_FLAGS})
    set_target_properties(comm_overlap_benchmark_cc PROPERTIES COMPILE_FLAGS ${CMAKE_COMPILE_FLAGS})
endif()

if (${CMAKE_LINK_FLAGS})
    set_target_properties(comm_overlap_benchmark_cc PROPERTIES LINK_FLAGS ${CMAKE_LINK_FLAGS}) 
endif()

target_link_libraries(
    comm_overlap_benchmark_cc
    ${MPI_LIBRARIES})

if (ENABLE_TIMER)
    target_link_libraries(comm_overlap_benchmark
        ${Boost_LIBRARIES}
    )
    target_link_libraries(comm_overlap_benchmark_cc
        ${Boost_LIBRARIES}
    )
endif()

install( TARGETS comm_overlap_benchmark DESTINATION bin/ )

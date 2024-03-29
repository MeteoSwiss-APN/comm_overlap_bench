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




#cuda and non cuda compilation
if( CUDA_BACKEND )
    cuda_add_executable(comm_overlap_benchmark
        ${SOURCES}
        ${CUDA_SOURCES}
        OPTIONS ${NVCC_FLAGS})
target_link_libraries(comm_overlap_benchmark CUDA::nvml) 
else()
    add_executable(comm_overlap_benchmark
        ${X86_SOURCES}
        ${SOURCES})
endif()

if (${CMAKE_COMPILE_FLAGS})
    set_target_properties(comm_overlap_benchmark PROPERTIES COMPILE_FLAGS ${CMAKE_COMPILE_FLAGS})
endif()

if (${CMAKE_LINK_FLAGS})
    set_target_properties(comm_overlap_benchmark PROPERTIES LINK_FLAGS ${CMAKE_LINK_FLAGS}) 
endif()

if (ENABLE_TIMER)
    target_link_libraries(comm_overlap_benchmark
        ${Boost_LIBRARIES}
    )
endif()

install( TARGETS comm_overlap_benchmark DESTINATION bin/ )

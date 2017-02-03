set( 
    SOURCES
    "Definitions.h"
    "HoriDiffBenchmark.cpp"
    "HoriDiffReference.cpp"
#    "main.cpp"
    "Environment.h"
    "Environment.cpp"
    "IJKSize.h"
    "Kernel.h"
    "Repository.h"
    "Repository.cpp"
    "SimpleStorage.h"
    "SimpleStorage.cpp"
    "SimpleSwappableStorage.h"
    "SimpleSwappableStorage.cpp"
    "Options.h"
    "Options.cpp"
    "HorizontalDiffusionSA.h"
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
    cuda_add_executable(
        StandaloneStencilsCUDA
        ${CUDA_SOURCES} 
        ${SOURCES}
        OPTIONS "${HORIDIFF_NVCC_FLAGS}"
    )
    target_link_libraries( StandaloneStencilsCUDA ${Boost_LIBRARIES} ${MPI_LIBRARIES})

    if(MPI_COMPILE_FLAGS)
      set_target_properties(StandaloneStencilsCUDA PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    endif()

    if(MPI_LINK_FLAGS)
      set_target_properties(StandaloneStencilsCUDA PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
    endif()


    install( TARGETS StandaloneStencilsCUDA DESTINATION bin/ )
else()
    add_executable(
        StandaloneStencils 
        ${X86_SOURCES}
        ${SOURCES}
    )
    target_link_libraries( StandaloneStencils ${MPI_LIBRARIES})

    if(MPI_COMPILE_FLAGS)
      set_target_properties(StandaloneStencils PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    endif()

    if(MPI_LINK_FLAGS)
      set_target_properties(StandaloneStencils PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
    endif()

    install( TARGETS StandaloneStencils DESTINATION bin/ )
 
endif()



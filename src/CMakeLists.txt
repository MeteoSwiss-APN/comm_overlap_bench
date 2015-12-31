set( 
    SOURCES
    "CacheFlush.h"
    "HoriDiffRepository.h"
    "HoriDiffBenchmark.cpp"
    "HoriDiffRepository.cpp"
    "HoriDiffReference.cpp"
    "Options.cpp"
    "UnittestEnvironment.cpp"
    "TestMain.cpp"
    "HorizontalDiffusionSA.h"
    "UnittestEnvironment.h"
    "Options.h"
) 

set(
    X86_SOURCES
    "HorizontalDiffusionSA.cpp"
)

set(
    CUDA_SOURCES
    "HorizontalDiffusionSA.cu"
)

#cuda and non cuda compilation
if( CUDA_BACKEND )
    cuda_add_executable(
        StandaloneStencilsCUDA
        ${CUDA_SOURCES} 
        ${SOURCES}
        OPTIONS "${HORIDIFF_NVCC_FLAGS}"
    )
    target_link_libraries( StandaloneStencilsCUDA ${STELLA_LIBRARIES} )

    install( TARGETS StandaloneStencilsCUDA DESTINATION bin/ )
else()
    add_executable(
        StandaloneStencils 
        ${X86_SOURCES}
        ${SOURCES}
    )
    target_link_libraries( StandaloneStencils ${STELLA_LIBRARIES} )

    install( TARGETS StandaloneStencils DESTINATION bin/ )
 
endif()



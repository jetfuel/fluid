if(WITH_GPU)
    add_definitions(-DPADDLE_WITH_CUDA)
    add_definitions(-DPADDLE_USE_DSO) # Compile PaddlePaddle with dynamic linked CUDA

    FIND_PACKAGE(CUDA REQUIRED)

    if(${CUDA_VERSION_MAJOR} VERSION_LESS 7)
        message(FATAL_ERROR "Paddle needs CUDA >= 7.0 to compile")
    endif()

    include(find_cudnn)
    if(NOT CUDNN_FOUND)
        message(FATAL_ERROR "Paddle needs cudnn to compile")
    endif()
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler ${SIMD_FLAG}")

    # Include cuda and cudnn
    include_directories(${CUDNN_INCLUDE_DIR})
    include_directories(${CUDA_TOOLKIT_INCLUDE})
    # add cudnn library
    include(external/cudnn)
endif()

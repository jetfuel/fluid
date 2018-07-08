cc_library(dynamic_loader SRCS dynamic_loader.cc DEPS glog gflags enforce)

list(APPEND CUDA_SRCS cublas.cc cudnn.cc curand.cc)

# There is no macOS version of NCCL.
if (NOT APPLE)
  list(APPEND CUDA_SRCS nccl.cc)
endif()

nv_library(dynload_cuda SRCS ${CUDA_SRCS} DEPS dynamic_loader)

cc_library(enforce SRCS enforce.cc DEPS glog)
cc_test(enforce_test SRCS enforce_test.cc DEPS stringpiece enforce gtest)

cc_library(cpu_info SRCS cpu_info.cc DEPS gflags glog enforce)
cc_test(cpu_info_test SRCS cpu_info_test.cc DEPS cpu_info gtest)

nv_library(gpu_info SRCS gpu_info.cc DEPS gflags glog enforce)

cc_library(place SRCS place.cc DEPS enforce boost)
cc_test(place_test SRCS place_test.cc DEPS place glog gflags gtest)

add_subdirectory(dynload)

IF(WITH_GPU)
    set(GPU_CTX_DEPS dynload_cuda dynamic_loader)
ELSE()
    set(GPU_CTX_DEPS)
ENDIF()

# memcpy depends on device_context, here add deps individually for
# avoiding cycle dependencies
cc_library(device_context SRCS device_context.cc DEPS malloc
  place eigen3 ${GPU_CTX_DEPS})

nv_test(device_context_test SRCS device_context_test.cu DEPS device_context gpu_info gtest)

nv_test(cudnn_helper_test SRCS cudnn_helper_test.cc DEPS dynload_cuda cudnn gtest)
nv_test(transform_test SRCS transform_test.cu DEPS memory place device_context gtest)

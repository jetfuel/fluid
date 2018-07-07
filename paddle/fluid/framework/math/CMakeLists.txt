if (WITH_GPU)
    nv_library(math_function SRCS math_function.cc math_function.cu DEPS enforce eigen3 tensor device_context cblas framework_proto)
else()
    cc_library(math_function SRCS math_function.cc DEPS enforce eigen3 tensor device_context cblas framework_proto)
endif()

cc_library(blas SRCS blas.cc DEPS cblas framework_proto device_context)
cc_test(math_function_test SRCS math_function_test.cc DEPS math_function device_context tensor blas)
nv_test(math_function_gpu_test SRCS math_function_test.cu DEPS math_function device_context tensor blas)

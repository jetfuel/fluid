cc_library(ddim SRCS ddim.cc DEPS enforce eigen3 boost)
cc_test(ddim_test SRCS ddim_test.cc DEPS ddim)
nv_test(dim_test SRCS dim_test.cu DEPS ddim)

proto_library(framework_proto SRCS framework.proto)

cc_library(data_type SRCS data_type.cc DEPS framework_proto ddim device_context)

if(WITH_GPU)
  nv_library(tensor SRCS tensor.cc tensor_util.cu DEPS place memory data_type)
else()
  cc_library(tensor SRCS tensor.cc tensor_util.cc DEPS place memory data_type)
endif()

cc_test(tensor_test SRCS tensor_test.cc DEPS tensor)

if(WITH_GPU)
  nv_test(tensor_util_test SRCS tensor_util_test.cc tensor_util_test.cu DEPS tensor)
else()
  cc_test(tensor_util_test SRCS tensor_util_test.cc DEPS tensor)
endif()

cc_test(eigen_test SRCS eigen_test.cc DEPS tensor)

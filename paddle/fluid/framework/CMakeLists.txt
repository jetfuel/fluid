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

nv_test(vector_test SRCS vector_test.cu DEPS place memory device_context)

cc_library(lod_tensor SRCS lod_tensor.cc DEPS ddim place tensor framework_proto recordio)
cc_test(lod_tensor_test SRCS lod_tensor_test.cc DEPS lod_tensor memory)
nv_test(lod_tensor_gpu_test SRCS lod_tensor_test.cu DEPS lod_tensor)

cc_test(variable_test SRCS variable_test.cc)

cc_library(scope SRCS scope.cc DEPS glog)
cc_test(scope_test SRCS scope_test.cc DEPS scope)

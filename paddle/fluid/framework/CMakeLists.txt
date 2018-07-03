cc_library(ddim SRCS ddim.cc DEPS enforce eigen3 boost)
cc_test(ddim_test SRCS ddim_test.cc DEPS ddim)
nv_test(dim_test SRCS dim_test.cu DEPS ddim)

proto_library(framework_proto SRCS framework.proto)

cc_library(data_type SRCS data_type.cc DEPS framework_proto ddim device_context)

cc_library(tensor_data_layout SRCS tensor_data_layout.cc DEPS enforce)

if(WITH_GPU)
  nv_library(tensor SRCS tensor.cc tensor_util.cu DEPS enforce tensor_data_layout place memory data_type)
else()
  cc_library(tensor SRCS tensor.cc tensor_util.cc DEPS enforce tensor_data_layout place memory data_type)
endif()

cc_test(tensor_test SRCS tensor_test.cc DEPS tensor)

if(WITH_GPU)
  nv_test(tensor_util_test SRCS tensor_util_test.cc tensor_util_test.cu DEPS tensor)
else()
  cc_test(tensor_util_test SRCS tensor_util_test.cc DEPS tensor)
endif()

cc_test(eigen_test SRCS eigen_test.cc DEPS tensor)

nv_test(vector_test SRCS vector_test.cu DEPS place memory device_context)

cc_library(lod_tensor SRCS lod_tensor.cc DEPS enforce ddim place tensor framework_proto recordio)
cc_test(lod_tensor_test SRCS lod_tensor_test.cc DEPS lod_tensor memory)
nv_test(lod_tensor_gpu_test SRCS lod_tensor_test.cu DEPS lod_tensor)

cc_library(selected_rows SRCS selected_rows.cc DEPS enforce tensor)
cc_test(selected_rows_test SRCS selected_rows_test.cc DEPS selected_rows)

cc_library(variable SRCS variable.cc DEPS enforce)
cc_test(variable_test SRCS variable_test.cc DEPS variable)

cc_library(scope SRCS scope.cc DEPS enforce)
cc_test(scope_test SRCS scope_test.cc DEPS scope)

cc_library(accelerator SRCS accelerator.cc DEPS enforce)
cc_test(accelerator_test SRCS accelerator_test.cc DEPS accelerator)

cc_library(op_kernel_type SRCS op_kernel_type.cc DEPS place device_context data_type tensor_data_layout accelerator framework_proto)
cc_test(op_kernel_type_test SRCS op_kernel_type_test.cc DEPS op_kernel_type)

cc_library(attribute SRCS attribute.cc DEPS framework_proto boost)

cc_library(op_proto_maker SRCS op_proto_maker.cc DEPS attribute framework_proto)
cc_test(op_proto_maker_test SRCS op_proto_maker_test.cc DEPS op_proto_maker enforce)

cc_library(op_desc SRCS op_desc.cc DEPS op_proto_maker)

cc_library(grad_op_desc_maker SRCS grad_op_desc_maker.cc DEPS op_desc)

cc_library(reader SRCS reader.cc DEPS lod_tensor ddim)

# cc_library(op_info SRCS op_info.cc DEPS attribute framework_proto)

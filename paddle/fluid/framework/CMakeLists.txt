
cc_library(ddim SRCS ddim.cc DEPS enforce eigen3 boost)
cc_test(ddim_test SRCS ddim_test.cc DEPS ddim)
nv_test(dim_test SRCS dim_test.cu DEPS ddim)

proto_library(framework_proto SRCS framework.proto)

cc_library(data_type SRCS data_type.cc DEPS framework_proto)

cc_library(tensor_data_layout SRCS tensor_data_layout.cc DEPS enforce)

if(WITH_GPU)
  nv_library(tensor SRCS tensor.cc tensor_util.cu DEPS ddim enforce tensor_data_layout place memory data_type device_context)
else()
  cc_library(tensor SRCS tensor.cc tensor_util.cc DEPS ddim enforce tensor_data_layout place memory data_type device_context)
endif()

cc_test(tensor_test SRCS tensor_test.cc DEPS tensor)

if(WITH_GPU)
  nv_test(tensor_util_test SRCS tensor_util_test.cc tensor_util_test.cu DEPS tensor)
else()
  cc_test(tensor_util_test SRCS tensor_util_test.cc DEPS tensor)
endif()

cc_test(eigen_test SRCS eigen_test.cc DEPS tensor)

nv_test(vector_test SRCS vector_test.cu DEPS place memory device_context tensor)

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

cc_library(op_kernel_type SRCS op_kernel_type.cc DEPS place data_type tensor_data_layout accelerator framework_proto)
cc_test(op_kernel_type_test SRCS op_kernel_type_test.cc DEPS op_kernel_type)

cc_library(attribute SRCS attribute.cc DEPS framework_proto boost)

cc_library(op_proto_maker SRCS op_proto_maker.cc DEPS attribute framework_proto)
cc_test(op_proto_maker_test SRCS op_proto_maker_test.cc DEPS op_proto_maker enforce)

cc_library(op_desc SRCS op_desc.cc DEPS op_proto_maker)

cc_library(grad_op_desc_maker SRCS grad_op_desc_maker.cc DEPS op_desc)


if(WITH_GPU)
  nv_library(data_type_transform SRCS data_type_transform.cu DEPS lod_tensor op_kernel_type)
  nv_test(data_type_transform_test SRCS data_type_transform_test.cc data_type_transform_test.cu DEPS data_type_transform framework_proto tensor_data_layout accelerator)
else()
  cc_library(data_type_transform SRCS data_type_transform.cc DEPS lod_tensor op_kernel_type)
  cc_test(data_type_transform_test SRCS data_type_transform_test.cc DEPS data_type_transform framework_proto tensor_data_layout accelerator)
endif()

cc_library(data_device_transform SRCS data_device_transform.cc DEPS lod_tensor enforce)
# TODO(yi): The current unit test is too much heavier than needed and
# is out of the scope of a "unit test". We need to write a new one.

# math is used in data_layout_transform
add_subdirectory(math)
cc_library(data_layout_transform SRCS data_layout_transform.cc DEPS lod_tensor enforce op_kernel_type device_context tensor_data_layout math_function)
cc_test(data_layout_transform_test SRCS data_layout_transform_test.cc DEPS data_layout_transform)

cc_library(data_transform SRCS data_transform.cc DEPS enforce op_kernel_type lod_tensor variable selected_rows data_device_transform data_type_transform data_layout_transform)

cc_library(reader SRCS reader.cc DEPS lod_tensor ddim)

cc_library(shape_inference SRCS shape_inference.cc DEPS ddim attribute op_desc grad_op_desc_maker variable)

cc_library(operator SRCS operator.cc DEPS device_context variable shape_inference lod_tensor scope glog data_transform enforce accelerator)
# TODO(tonyyang-svail): make operator test lighter, current one depends on op_registry

cc_library(op_info SRCS op_info.cc DEPS attribute)

cc_library(op_registry SRCS op_registry.cc DEPS op_proto_maker op_info operator glog op_desc)
nv_test(op_registry_test SRCS op_registry_test.cc DEPS op_registry)


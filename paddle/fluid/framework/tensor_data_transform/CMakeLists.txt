cc_library(tensor_data_transform_device SRCS device.cc DEPS tensor)
nv_test(tensor_data_transform_device_test SRCS device_test.cu DEPS operator op_registry) # init math_function)

if(WITH_GPU)
  nv_library(tensor_data_transform_type SRCS type.cu DEPS tensor)
  nv_test(tensor_data_transform_type_test SRCS type_test.cc type_test.cu DEPS tensor_data_transform_type)
else()
  cc_library(tensor_data_transform_type SRCS type.cc DEPS tensor)
  cc_test(tensor_data_transform_type_test SRCS type_test.cc DEPS tensor_data_transform_type)
endif()

cc_library(tensor_data_transform_layout SRCS layout.cc DEPS tensor) # math_function)
cc_test(tensor_data_transform_layout_test SRCS layout_test.cc DEPS tensor_data_transform_layout)

cc_library(tensor_data_transform SRCS tensor_data_transform.cc DEPS # math_function
  framework_proto
  tensor
  selected_rows
  tensor_data_transform_device
  tensor_data_transform_type
  tensor_data_transform_layout)

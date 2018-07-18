/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/framework/data_type_transform.h"

#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_data_layout.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/transform.h"

TEST(DataTypeTransform, GPUTransform) {
  auto cpu_place = paddle::fluid::platform::CPUPlace();
  auto gpu_place = paddle::fluid::platform::CUDAPlace(0);
  paddle::fluid::platform::CUDADeviceContext context(gpu_place);

  auto kernel_fp32 = paddle::fluid::framework::OpKernelType(
      paddle::fluid::framework::proto::VarType::FP32,
      gpu_place,
      paddle::fluid::framework::TensorDataLayout::kAnyLayout,
      paddle::fluid::framework::Accelerator::kPlain);

  auto kernel_fp64 = paddle::fluid::framework::OpKernelType(
      paddle::fluid::framework::proto::VarType::FP64,
      gpu_place,
      paddle::fluid::framework::TensorDataLayout::kAnyLayout,
      paddle::fluid::framework::Accelerator::kPlain);

  auto kernel_int32 = paddle::fluid::framework::OpKernelType(
      paddle::fluid::framework::proto::VarType::INT32,
      gpu_place,
      paddle::fluid::framework::TensorDataLayout::kAnyLayout,
      paddle::fluid::framework::Accelerator::kPlain);

  auto kernel_int64 = paddle::fluid::framework::OpKernelType(
      paddle::fluid::framework::proto::VarType::INT64,
      gpu_place,
      paddle::fluid::framework::TensorDataLayout::kAnyLayout,
      paddle::fluid::framework::Accelerator::kPlain);

  auto kernel_bool = paddle::fluid::framework::OpKernelType(
      paddle::fluid::framework::proto::VarType::BOOL,
      gpu_place,
      paddle::fluid::framework::TensorDataLayout::kAnyLayout,
      paddle::fluid::framework::Accelerator::kPlain);

  // data type transform from float32
  {
    paddle::fluid::framework::Tensor in;
    paddle::fluid::framework::Tensor in_gpu;
    paddle::fluid::framework::Tensor out_gpu;
    paddle::fluid::framework::Tensor out;

    float* in_ptr = in.mutable_data<float>(
        paddle::fluid::framework::make_ddim({2, 3}), cpu_place);
    float arr[6] = {0, 1, 2, 3, 4, 5};
    int data_number = sizeof(arr) / sizeof(arr[0]);
    memcpy(in_ptr, arr, sizeof(arr));

    paddle::fluid::framework::TensorCopy(in, gpu_place, context, &in_gpu);
    context.Wait();
    paddle::fluid::framework::TransDataType(
        kernel_fp32, kernel_fp64, in_gpu, &out_gpu);
    paddle::fluid::framework::TensorCopy(out_gpu, cpu_place, context, &out);
    context.Wait();

    double* out_data_double = out.data<double>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_double[i], static_cast<double>(arr[i]));
    }

    paddle::fluid::framework::TransDataType(
        kernel_fp32, kernel_int32, in_gpu, &out_gpu);
    paddle::fluid::framework::TensorCopy(out_gpu, cpu_place, context, &out);
    context.Wait();

    int* out_data_int = out.data<int>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_int[i], static_cast<int>(arr[i]));
    }
  }
}

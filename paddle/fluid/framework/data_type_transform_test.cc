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
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/transform.h"

TEST(DataTypeTransform, CPUTransform) {
  auto place = paddle::fluid::platform::CPUPlace();

  auto kernel_fp16 = paddle::fluid::framework::OpKernelType(
      paddle::fluid::framework::proto::VarType::FP16, place,
      paddle::fluid::framework::TensorDataLayout::kAnyLayout,
      paddle::fluid::framework::Accelerator::kPlain);

  auto kernel_fp32 = paddle::fluid::framework::OpKernelType(
      paddle::fluid::framework::proto::VarType::FP32, place,
      paddle::fluid::framework::TensorDataLayout::kAnyLayout,
      paddle::fluid::framework::Accelerator::kPlain);

  auto kernel_fp64 = paddle::fluid::framework::OpKernelType(
      paddle::fluid::framework::proto::VarType::FP64, place,
      paddle::fluid::framework::TensorDataLayout::kAnyLayout,
      paddle::fluid::framework::Accelerator::kPlain);

  auto kernel_int32 = paddle::fluid::framework::OpKernelType(
      paddle::fluid::framework::proto::VarType::INT32, place,
      paddle::fluid::framework::TensorDataLayout::kAnyLayout,
      paddle::fluid::framework::Accelerator::kPlain);

  auto kernel_int64 = paddle::fluid::framework::OpKernelType(
      paddle::fluid::framework::proto::VarType::INT64, place,
      paddle::fluid::framework::TensorDataLayout::kAnyLayout,
      paddle::fluid::framework::Accelerator::kPlain);

  auto kernel_bool = paddle::fluid::framework::OpKernelType(
      paddle::fluid::framework::proto::VarType::BOOL, place,
      paddle::fluid::framework::TensorDataLayout::kAnyLayout,
      paddle::fluid::framework::Accelerator::kPlain);

  // data type transform from float32
  {
    paddle::fluid::framework::Tensor in;
    paddle::fluid::framework::Tensor out;

    float* ptr =
        in.mutable_data<float>(paddle::fluid::framework::make_ddim({2, 3}), place);
    int data_number = 2 * 3;

    for (int i = 0; i < data_number; ++i) {
      ptr[i] = i / 3;
    }

    paddle::fluid::framework::TransDataType(kernel_fp32, kernel_fp64, in, &out);
    double* out_data_double = out.data<double>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_double[i], static_cast<double>(i / 3));
    }

    paddle::fluid::framework::TransDataType(kernel_fp32, kernel_int32, in, &out);
    int* out_data_int = out.data<int>();
    for (int i = 0; i < data_number; ++i) {
      EXPECT_EQ(out_data_int[i], static_cast<int>(i / 3));
    }
  }
}

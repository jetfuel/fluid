/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_kernel_type.h"
#include <gtest/gtest.h>

TEST(OpKernelType, ToString) {
  using OpKernelType = paddle::fluid::framework::OpKernelType;
  using DataType = paddle::fluid::framework::proto::VarType;
  using CPUPlace = paddle::fluid::platform::CPUPlace;
  using TensorDataLayout = paddle::fluid::framework::TensorDataLayout;
  using Accelerator = paddle::fluid::framework::Accelerator;

  OpKernelType op_kernel_type(DataType::FP32, CPUPlace(), TensorDataLayout::kNCHW,
                              Accelerator::kCUDNN);

  ASSERT_EQ(paddle::fluid::framework::KernelTypeToString(op_kernel_type),
            "data_type[float]:data_layout[NCHW]:place[CPUPlace]:accelerator["
            "CUDNN]");
}

TEST(OpKernelType, Hash) {
  using OpKernelType = paddle::fluid::framework::OpKernelType;
  using DataType = paddle::fluid::framework::proto::VarType;
  using CPUPlace = paddle::fluid::platform::CPUPlace;
  using CUDAPlace = paddle::fluid::platform::CUDAPlace;
  using TensorDataLayout = paddle::fluid::framework::TensorDataLayout;
  using Accelerator = paddle::fluid::framework::Accelerator;

  OpKernelType op_kernel_type_1(DataType::FP32, CPUPlace(), TensorDataLayout::kNCHW,
                                Accelerator::kCUDNN);
  OpKernelType op_kernel_type_2(DataType::FP32, CUDAPlace(0), TensorDataLayout::kNCHW,
                                Accelerator::kCUDNN);

  OpKernelType::Hash hasher;
  ASSERT_NE(hasher(op_kernel_type_1), hasher(op_kernel_type_2));
}

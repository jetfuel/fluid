//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "paddle/fluid/framework/data_layout_transform.h"

#include "gtest/gtest.h"
#include "paddle/fluid/framework/math/math_function.h"
#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_data_layout.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"

TEST(DataTransform, TensorDataLayoutFunction) {
  auto place = paddle::fluid::platform::CPUPlace();
  paddle::fluid::framework::Tensor in = paddle::fluid::framework::Tensor();
  paddle::fluid::framework::Tensor out = paddle::fluid::framework::Tensor();
  in.mutable_data<double>(paddle::fluid::framework::make_ddim({2, 3, 1, 2}),
                          place);
  in.set_layout(paddle::fluid::framework::TensorDataLayout::kNHWC);

  auto kernel_nhwc = paddle::fluid::framework::OpKernelType(
      paddle::fluid::framework::proto::VarType::FP32,
      place,
      paddle::fluid::framework::TensorDataLayout::kNHWC,
      paddle::fluid::framework::Accelerator::Type::kPlain);
  auto kernel_ncwh = paddle::fluid::framework::OpKernelType(
      paddle::fluid::framework::proto::VarType::FP32,
      place,
      paddle::fluid::framework::TensorDataLayout::kNCHW,
      paddle::fluid::framework::Accelerator::Type::kPlain);

  paddle::fluid::framework::TransDataLayout(kernel_nhwc, kernel_ncwh, in, &out);

  EXPECT_TRUE(out.layout() ==
              paddle::fluid::framework::TensorDataLayout::kNCHW);
  EXPECT_TRUE(out.dims() == paddle::fluid::framework::make_ddim({2, 2, 3, 1}));

  TransDataLayout(kernel_ncwh, kernel_nhwc, in, &out);

  EXPECT_TRUE(in.layout() == paddle::fluid::framework::TensorDataLayout::kNHWC);
  EXPECT_TRUE(in.dims() == paddle::fluid::framework::make_ddim({2, 3, 1, 2}));
}

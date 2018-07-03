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

#include <map>
#include <vector>

#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_data_layout.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
// #include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace fluid {
namespace framework {

namespace {

std::vector<int> GetAxis(const TensorDataLayout& from, const TensorDataLayout& to) {
  PADDLE_ENFORCE_NE(from, to,
                    "layout transform should transform different layout");
  if (from == TensorDataLayout::kNCHW && to == TensorDataLayout::kNHWC) {
    return {0, 2, 3, 1};
  } else if (from == TensorDataLayout::kNHWC && to == TensorDataLayout::kNCHW) {
    return {0, 3, 1, 2};
  } else {
    PADDLE_THROW("unsupported transform");
  }
}

struct CastTensorDataLayout {
  CastTensorDataLayout(const platform::DeviceContext* ctx,
                 const std::vector<int>& axis, const framework::Tensor& in,
                 framework::Tensor* out)
      : in_(in), out_(out), ctx_(ctx), axis_(axis) {}
  const framework::Tensor in_;
  framework::Tensor* out_;
  const platform::DeviceContext* ctx_;
  const std::vector<int> axis_;

  template <typename T>
  void operator()() {
    // TODO(yi): Add the implementation back after porting
    // operators/math/math_function.h to framework/tensor_math.h.

    // auto place = ctx_->GetPlace();
    // if (platform::is_cpu_place(place)) {
    //   operators::math::Transpose<platform::CPUDeviceContext, T, 4> trans4;
    //   auto* context = static_cast<const platform::CPUDeviceContext*>(ctx_);
    //   trans4(*context, in_, out_, axis_);
    // } else {
    //   PADDLE_THROW("Unsupport CPU <-> GPU!");
    // }
    PADDLE_THROW("CastTensorDataLayout not yet implemented");
  }
};

}  // namespace

void TransDataLayout(const OpKernelType& kernel_type_for_var,
                     const OpKernelType& expected_kernel_type, const Tensor& in,
                     Tensor* out) {
  // TODO(yi): Add the implementation back after porting
  // operators/math/math_function.h to framework/tensor_math.h.

  // PADDLE_ENFORCE(
  //     platform::places_are_same_class(kernel_type_for_var.place_,
  //                                     expected_kernel_type.place_),
  //     "TransDataLayout only support DataLayout transform on same place!");

  // PADDLE_ENFORCE(arity(in.dims()) == 4, "Input Arity only support 4!");

  // auto& pool = platform::DeviceContextPool::Instance();

  // auto src_dim = in.dims();
  // std::vector<int64_t> dst_dim;

  // auto axis = GetAxis(kernel_type_for_var.data_layout_,
  //                     expected_kernel_type.data_layout_);
  // dst_dim.resize(axis.size());
  // for (size_t i = 0; i < axis.size(); i++) {
  //   dst_dim[i] = src_dim[axis[i]];
  // }

  // out->Resize(make_ddim(dst_dim));
  // out->mutable_data(expected_kernel_type.place_, in.type());

  // framework::VisitDataType(
  //     framework::ToDataType(in.type()),
  //     CastTensorDataLayout(pool.Get(expected_kernel_type.place_), axis, in, out));

  // out->set_layout(expected_kernel_type.data_layout_);
  PADDLE_THROW("TransDataLayout not yet implemented");
}

}  // namespace framework
}  // namespace fluid
}  // namespace paddle

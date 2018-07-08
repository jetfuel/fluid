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

#pragma once

#include <string>
#include "paddle/fluid/framework/tensor_data_layout.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/accelerator.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace fluid {
namespace framework {

struct OpKernelType {
  struct Hash {
    size_t operator()(const OpKernelType& key) const;
  };

  // place, data_type, accelerator kinds less than 2^8
  constexpr static int LEFT_SHIFT = 8;

  proto::VarType::Type data_type_;
  TensorDataLayout data_layout_;
  platform::Place place_;
  Accelerator accelerator_;

  OpKernelType(proto::VarType::Type data_type, platform::Place place,
               TensorDataLayout data_layout = TensorDataLayout::kAnyLayout,
               Accelerator accelerator = Accelerator::kPlain);

  OpKernelType(proto::VarType::Type data_type,
               const platform::DeviceContext& dev_ctx,
               TensorDataLayout data_layout = TensorDataLayout::kAnyLayout,
               Accelerator accelerator = Accelerator::kPlain);

  bool operator==(const OpKernelType& o) const {
    return platform::places_are_same_class(place_, o.place_) &&
           data_type_ == o.data_type_ && data_layout_ == o.data_layout_ &&
           accelerator_ == o.accelerator_;
  }

  bool operator!=(const OpKernelType& o) const { return !(*this == o); }
};

std::ostream& operator<<(std::ostream& os, const OpKernelType& kernel_key);
std::string KernelTypeToString(const OpKernelType& kernel_key);
bool NeedTransformLayout(const TensorDataLayout& l, const TensorDataLayout& r);
bool NeedTransform(const OpKernelType& l, const OpKernelType& r);

}  // namespace framework
}  // namespace fluid
}  // namespace paddle

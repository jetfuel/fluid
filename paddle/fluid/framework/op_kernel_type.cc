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

namespace paddle {
namespace fluid {
namespace framework {

size_t OpKernelType::Hash::operator()(const OpKernelType& key) const {
  int place = key.place_.which();
  int data_type = static_cast<int>(key.data_type_) << LEFT_SHIFT;
  int data_layout = static_cast<int>(key.data_layout_.type_) << (LEFT_SHIFT * 2);
  int accelerator = static_cast<int>(key.accelerator_.type_)
                    << (LEFT_SHIFT * 3);

  std::hash<int> hasher;
  return hasher(place + data_type + data_layout + accelerator);
}

OpKernelType::OpKernelType(proto::VarType::Type data_type, platform::Place place,
                           TensorDataLayout data_layout,
                           Accelerator accelerator)
    : data_type_(data_type),
      data_layout_(data_layout),
      place_(place),
      accelerator_(accelerator) {}

OpKernelType::OpKernelType(proto::VarType::Type data_type,
                           const platform::DeviceContext& dev_ctx,
                           TensorDataLayout data_layout,
                           Accelerator accelerator)
    : data_type_(data_type),
      data_layout_(data_layout),
      place_(dev_ctx.GetPlace()),
      accelerator_(accelerator) {}

std::ostream& operator<<(std::ostream& os, const OpKernelType& kernel_key) {
  os << "data_type[" << kernel_key.data_type_ << "]:data_layout["
     << kernel_key.data_layout_ << "]:place[" << kernel_key.place_
     << "]:accelerator[" << kernel_key.accelerator_ << "]";
  return os;
}

std::string KernelTypeToString(const OpKernelType& kernel_key) {
  std::ostringstream stream;
  stream << kernel_key;
  return stream.str();
}

bool NeedTransformLayout(const TensorDataLayout& l, const TensorDataLayout& r) {
  bool ret =
      (l != TensorDataLayout::kAnyLayout && r != TensorDataLayout::kAnyLayout && l != r);
  return ret;
}

bool NeedTransform(const OpKernelType& l, const OpKernelType& r) {
  return (!platform::places_are_same_class(l.place_, r.place_)) ||
      (l.data_type_ != r.data_type_) ||
      NeedTransformLayout(l.data_layout_, r.data_layout_);
}

}  // namespace framework
}  // namespace fluid
}  // namespace paddle

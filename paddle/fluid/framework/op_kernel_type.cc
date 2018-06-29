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
#ifdef PADDLE_WITH_MKLDNN
  // Layout transform needed for either non-MKLDNN to MKLDNN or vice versa
  ret |= (l != TensorDataLayout::kMKLDNN && r == TensorDataLayout::kMKLDNN);
  ret |= (l == TensorDataLayout::kMKLDNN && r != TensorDataLayout::kMKLDNN);
#endif
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

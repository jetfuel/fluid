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
#include "paddle/fluid/framework/tensor_data_layout.h"

#include <algorithm>
#include <cctype>
#include <ostream>
#include <string>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace fluid {
namespace framework {

TensorDataLayout::TensorDataLayout(const std::string& type) {
  std::string key(type);
  std::transform(key.begin(), key.end(), key.begin(), ::toupper);

  std::vector<std::string> layouts = {
      "NHWC", "NCHW", "ANYLAYOUT", "MKLDNNLAYOUT"};
  for (int i = 0; i < layouts.size(); ++i) {
    if (key == layouts[i]) {
      type_ = static_cast<TensorDataLayout::Type>(i);
      return;
    }
  }
  PADDLE_THROW("Unknown TensorDataLayout %s", type);
}

std::string TensorDataLayout::ToString() const {
  std::vector<std::string> layouts = {
      "NHWC", "NCHW", "ANYLAYOUT", "MKLDNNLAYOUT"};
  return layouts[type_];
}

std::ostream& operator<<(std::ostream& out, const TensorDataLayout& l) {
  out << l.ToString();
  return out;
}

}  // namespace framework
}  // namespace fluid
}  // namespace paddle

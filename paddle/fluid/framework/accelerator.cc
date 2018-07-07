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
#include "paddle/fluid/framework/accelerator.h"

#include <algorithm>  // for transform
#include <cctype>  // for toupper
#include <string>
#include <vector>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace fluid {
namespace framework {

Accelerator::Accelerator(const char* type) {
  std::string key(type);
  std::transform(key.begin(), key.end(), key.begin(), ::toupper);

  std::vector<std::string> accelerators = {"PLAIN", "MKLDNN", "CUDNN"};
  std::vector<std::string> non_accelerators = {"CPU", "CUDA"};

  for (int i = 0; i < accelerators.size(); ++i) {
    if (key == accelerators[i]) {
      type_ = static_cast<Accelerator::Type>(i);
      return;
    }
  }
  for (auto& plain : non_accelerators) {
    if (key == plain) {
      type_ = static_cast<Accelerator::Type>(0);
      return;
    }
  }
  PADDLE_THROW("Unknown Accelerator %s", type);
}

std::string Accelerator::ToString() const {
  std::vector<std::string> accelerators = {"PLAIN", "MKLDNN", "CUDNN"};
  return accelerators[type_];
}

std::ostream& operator<<(std::ostream& out, Accelerator l) {
  out << l.ToString();
  return out;
}

}  // namespace framework
}  // namespace fluid
}  // namespace paddle

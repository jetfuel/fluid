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
namespace framework {

namespace {

std::string toupper(const char* s) {
  std::string r(s);
  std::transform(r.begin(), r.end(), r.begin(), ::toupper);
  return r;
}

std::vector<std::string> accelerators = {"PLAIN", "MKLDNN", "CUDNN"};

}  // namespace

Accelerator::Accelerator(const char* type) {
  std::string key = toupper(type);
  for (int i = 0; i < accelerators.size(); ++i) {
    if (key == accelerators[i]) {
      type_ = static_cast<Accelerator::Type>(i);
      return;
    }
  }
  PADDLE_THROW("Unknown Accelerator %s", type);
}

std::string Accelerator::ToString() const {
  return accelerators[type_];
}

std::ostream& operator<<(std::ostream& out, Accelerator l) {
  out << l.ToString();
  return out;
}

}  // namespace framework
}  // namespace paddle

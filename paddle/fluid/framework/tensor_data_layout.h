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

#include <cctype>
#include <ostream>
#include <string>

namespace paddle {
namespace fluid {
namespace framework {

class TensorDataLayout {
 public:
  enum Type {
    kNHWC = 0,
    kNCHW = 1,
    kAnyLayout = 2,
    kMKLDNN = 3  // all layouts supported by MKLDNN internally
  };

  TensorDataLayout(TensorDataLayout::Type type)
      : type_(type){};  // NOLINT: so could we compare.
  explicit TensorDataLayout(const std::string& type);
  std::string ToString() const;

  Type type_;

  bool operator==(const TensorDataLayout& l) const { return type_ == l.type_; }
  bool operator!=(const TensorDataLayout& l) const { return type_ != l.type_; }
};

std::ostream& operator<<(std::ostream& out, const TensorDataLayout& l);

}  // namespace framework
}  // namespace fluid
}  // namespace paddle

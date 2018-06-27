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

#include <iostream>
#include <string>

namespace paddle {
namespace framework {

// For more details about the design of Accelerator, Please refer to
// https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/operator_kernel_type.md#library
class Accelerator {
 public:
  enum Type {
    kPlain = 0,
    kMKLDNN = 1,
    kCUDNN = 2
  };

  explicit Accelerator(const char* type);
  std::string ToString() const;

  Type type_;
};

std::ostream& operator<<(std::ostream& out, Accelerator l);

}  // namespace framework
}  // namespace paddle

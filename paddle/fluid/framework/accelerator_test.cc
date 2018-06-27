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

#include <sstream>

#include "gtest/gtest.h"

TEST(Accelerator, Constructor) {
  // Test no exceptions.
  {
    paddle::framework::Accelerator a("plain");
  }
  {
     paddle::framework::Accelerator a("cuDNN");
  }
  {
    paddle::framework::Accelerator a("Mkldnn");
  }
}

TEST(Accelerator, ToString) {
  // Test no exceptions.
  {
    paddle::framework::Accelerator a("plain");
    EXPECT_EQ(a.ToString(), "PLAIN");
  }
  {
    paddle::framework::Accelerator a("cuDNN");
    EXPECT_EQ(a.ToString(), "CUDNN");
  }
  {
    paddle::framework::Accelerator a("Mkldnn");
    EXPECT_EQ(a.ToString(), "MKLDNN");
  }
}

TEST(Accelerator, StreamOutput) {
  {
    paddle::framework::Accelerator a("plain");
    std::stringstream ss;
    ss << a;
    EXPECT_EQ(ss.str(), "PLAIN");
  }
  {
    paddle::framework::Accelerator a("cuDNN");
    std::stringstream ss;
    ss << a;
    EXPECT_EQ(ss.str(), "CUDNN");
  }
  {
    paddle::framework::Accelerator a("Mkldnn");
    std::stringstream ss;
    ss << a;
    EXPECT_EQ(ss.str(), "MKLDNN");
  }

}

//  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "paddle/fluid/platform/place.h"
#include <sstream>
#include "gtest/gtest.h"

TEST(Place, Equality) {
  paddle::fluid::platform::CPUPlace cpu;
  paddle::fluid::platform::CUDAPlace g0(0), g1(1), gg0(0);

  EXPECT_EQ(cpu, cpu);
  EXPECT_EQ(g0, g0);
  EXPECT_EQ(g1, g1);
  EXPECT_EQ(g0, gg0);

  EXPECT_NE(g0, g1);

  EXPECT_TRUE(paddle::fluid::platform::places_are_same_class(g0, gg0));
  EXPECT_FALSE(paddle::fluid::platform::places_are_same_class(g0, cpu));
}

TEST(Place, Default) {
  using paddle::fluid::platform::CPUPlace;
  using paddle::fluid::platform::CUDAPlace;
  using paddle::fluid::platform::CUDAPinnedPlace;

  using paddle::fluid::platform::is_cpu_place;
  using paddle::fluid::platform::is_gpu_place;
  using paddle::fluid::platform::is_cuda_pinned_place;

  using paddle::fluid::platform::get_place;
  using paddle::fluid::platform::set_place;

  using paddle::fluid::platform::default_cpu;
  using paddle::fluid::platform::default_gpu;
  using paddle::fluid::platform::default_cuda_pinned;

  using paddle::fluid::platform::which_place;
  EXPECT_EQ(0, which_place(get_place()));

  EXPECT_TRUE(is_gpu_place(default_gpu()));
  EXPECT_TRUE(is_cpu_place(default_cpu()));
  EXPECT_TRUE(is_cuda_pinned_place(default_cuda_pinned()));

  EXPECT_TRUE(is_cpu_place(get_place()));
  set_place(CUDAPlace(1));
  EXPECT_TRUE(is_gpu_place(get_place()));

  set_place(CUDAPinnedPlace());
  EXPECT_TRUE(is_cuda_pinned_place(get_place()));
}

TEST(Place, Print) {
  {
    std::stringstream ss;
    ss << paddle::fluid::platform::CUDAPlace(1);
    EXPECT_EQ("CUDAPlace(1)", ss.str());
  }
  {
    std::stringstream ss;
    ss << paddle::fluid::platform::CPUPlace();
    EXPECT_EQ("CPUPlace", ss.str());
  }
}

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

#include <cblas.h>
#include <stdio.h>

#include "gtest/gtest.h"

TEST(openblas, ddot) {
  double m[2] = {1.0, 1.0};
  double n[2] = {2.0, 2.0};
  int result;

  result = cblas_ddot(2, m, 1, n, 1);
  ASSERT_EQ(result, 4);
}

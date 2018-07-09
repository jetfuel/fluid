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

#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace fluid {
namespace memory {

// None of template arguments DstPlace and SrcPlace could be
// CUDAPlace; otherwise, refer to the next function template.
void Copy(const platform::Place& dst_place, void* dst,
          const platform::Place& src_place, const void* src, size_t num);

#ifdef PADDLE_WITH_CUDA
// When any one of DstPlace or SrcPlace is CUDAPlace, users are allows
// to specify a CUDA stream.
void Copy(const platform::Place& dst_place, void* dst,
          const platform::Place& src_place, const void* src, size_t num,
          cudaStream_t stream);
#endif

}  // namespace memory
}  // namespace fluid
}  // namespace paddle

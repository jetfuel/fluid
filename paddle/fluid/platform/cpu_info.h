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

#include <stddef.h>

namespace paddle {
namespace fluid {
namespace platform {

//! Get the maximum allocation size for a machine.
size_t CpuMaxAllocSize();

//! Get the maximum allocation size for a machine.
size_t CUDAPinnedMaxAllocSize();

//! Get the minimum chunk size for buddy allocator.
size_t CpuMinChunkSize();

//! Get the maximum chunk size for buddy allocator.
size_t CpuMaxChunkSize();

//! Get the minimum chunk size for buddy allocator.
size_t CUDAPinnedMinChunkSize();

//! Get the maximum chunk size for buddy allocator.
size_t CUDAPinnedMaxChunkSize();

}  // namespace platform
}  // namespace fluid
}  // namespace paddle

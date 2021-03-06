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

#include "paddle/fluid/memory/malloc.h"

#include <unordered_map>

#include "gtest/gtest.h"
#include "paddle/fluid/memory/detail/memory_block.h"
#include "paddle/fluid/platform/cpu_info.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/place.h"

inline bool is_aligned(void const *p) {
  return 0 == (reinterpret_cast<uintptr_t>(p) & 0x3);
}

size_t align(size_t size, paddle::fluid::platform::CPUPlace place) {
  size += sizeof(paddle::fluid::memory::detail::MemoryBlock::Desc);
  size_t alignment = paddle::fluid::platform::CpuMinChunkSize();
  size_t remaining = size % alignment;
  return remaining == 0 ? size : size + (alignment - remaining);
}

TEST(BuddyAllocator, CPUAllocation) {
  void *p = nullptr;

  EXPECT_EQ(p, nullptr);

  paddle::fluid::platform::CPUPlace cpu;
  p = paddle::fluid::memory::Alloc(cpu, 4096);

  EXPECT_NE(p, nullptr);

  paddle::fluid::platform::Place place = cpu;
  EXPECT_EQ(paddle::fluid::memory::Used(cpu),
            paddle::fluid::memory::memory_usage(place));

  paddle::fluid::memory::Free(cpu, p);
}

TEST(BuddyAllocator, CPUMultAlloc) {
  paddle::fluid::platform::CPUPlace cpu;

  std::unordered_map<void *, size_t> ps;

  size_t total_size = paddle::fluid::memory::Used(cpu);
  EXPECT_EQ(total_size, 0UL);

  for (auto size :
       {0, 128, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304}) {
    ps[paddle::fluid::memory::Alloc(cpu, size)] = size;

    // Buddy Allocator doesn't manage too large memory chunk
    if (paddle::fluid::memory::Used(cpu) == total_size) continue;

    size_t aligned_size = align(size, cpu);
    total_size += aligned_size;
    EXPECT_EQ(total_size, paddle::fluid::memory::Used(cpu));
  }

  for (auto p : ps) {
    EXPECT_EQ(is_aligned(p.first), true);
    paddle::fluid::memory::Free(cpu, p.first);

    // Buddy Allocator doesn't manage too large memory chunk
    if (paddle::fluid::memory::Used(cpu) == total_size) continue;

    size_t aligned_size = align(p.second, cpu);
    total_size -= aligned_size;
    EXPECT_EQ(total_size, paddle::fluid::memory::Used(cpu));
  }
}

#ifdef PADDLE_WITH_CUDA

size_t align(size_t size, paddle::fluid::platform::CUDAPlace place) {
  size += sizeof(paddle::fluid::memory::detail::MemoryBlock::Desc);
  size_t alignment = paddle::fluid::platform::GpuMinChunkSize();
  size_t remaining = size % alignment;
  return remaining == 0 ? size : size + (alignment - remaining);
}

TEST(BuddyAllocator, GPUAllocation) {
  void *p = nullptr;

  EXPECT_EQ(p, nullptr);

  paddle::fluid::platform::CUDAPlace gpu(0);
  p = paddle::fluid::memory::Alloc(gpu, 4096);

  EXPECT_NE(p, nullptr);

  paddle::fluid::platform::Place place = gpu;
  EXPECT_EQ(paddle::fluid::memory::Used(gpu),
            paddle::fluid::memory::memory_usage(place));

  paddle::fluid::memory::Free(gpu, p);
}

TEST(BuddyAllocator, GPUMultAlloc) {
  paddle::fluid::platform::CUDAPlace gpu;

  std::unordered_map<void *, size_t> ps;

  size_t total_size = paddle::fluid::memory::Used(gpu);
  EXPECT_EQ(total_size, 0UL);

  for (auto size :
       {0, 128, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304}) {
    ps[paddle::fluid::memory::Alloc(gpu, size)] = size;

    // Buddy Allocator doesn't manage too large memory chunk
    if (paddle::fluid::memory::Used(gpu) == total_size) continue;

    size_t aligned_size = align(size, gpu);
    total_size += aligned_size;
    EXPECT_EQ(total_size, paddle::fluid::memory::Used(gpu));
  }

  for (auto p : ps) {
    EXPECT_EQ(is_aligned(p.first), true);
    paddle::fluid::memory::Free(gpu, p.first);

    // Buddy Allocator doesn't manage too large memory chunk
    if (paddle::fluid::memory::Used(gpu) == total_size) continue;

    size_t aligned_size = align(p.second, gpu);
    total_size -= aligned_size;
    EXPECT_EQ(total_size, paddle::fluid::memory::Used(gpu));
  }
}

size_t align(size_t size, paddle::fluid::platform::CUDAPinnedPlace place) {
  size += sizeof(paddle::fluid::memory::detail::MemoryBlock::Desc);
  size_t alignment = paddle::fluid::platform::CUDAPinnedMinChunkSize();
  size_t remaining = size % alignment;
  return remaining == 0 ? size : size + (alignment - remaining);
}

TEST(BuddyAllocator, CUDAPinnedAllocator) {
  void *p = nullptr;

  EXPECT_EQ(p, nullptr);

  paddle::fluid::platform::CUDAPinnedPlace cpu;
  p = paddle::fluid::memory::Alloc(cpu, 4096);

  EXPECT_NE(p, nullptr);

  paddle::fluid::platform::Place place = cpu;
  EXPECT_EQ(paddle::fluid::memory::Used(cpu),
            paddle::fluid::memory::memory_usage(place));

  paddle::fluid::memory::Free(cpu, p);
}

TEST(BuddyAllocator, CUDAPinnedMultAllocator) {
  paddle::fluid::platform::CUDAPinnedPlace cpu;

  std::unordered_map<void *, size_t> ps;

  size_t total_size = paddle::fluid::memory::Used(cpu);
  EXPECT_EQ(total_size, 0UL);

  for (auto size :
       {0, 128, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304}) {
    ps[paddle::fluid::memory::Alloc(cpu, size)] = size;

    // Buddy Allocator doesn't manage too large memory chunk
    if (paddle::fluid::memory::Used(cpu) == total_size) continue;

    size_t aligned_size = align(size, cpu);
    total_size += aligned_size;
    EXPECT_EQ(total_size, paddle::fluid::memory::Used(cpu));
  }

  for (auto p : ps) {
    EXPECT_EQ(is_aligned(p.first), true);
    paddle::fluid::memory::Free(cpu, p.first);

    // Buddy Allocator doesn't manage too large memory chunk
    if (paddle::fluid::memory::Used(cpu) == total_size) continue;

    size_t aligned_size = align(p.second, cpu);
    total_size -= aligned_size;
    EXPECT_EQ(total_size, paddle::fluid::memory::Used(cpu));
  }
}
#endif

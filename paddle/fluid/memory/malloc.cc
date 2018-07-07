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

#include "glog/logging.h"

#include "paddle/fluid/memory/detail/buddy_allocator.h"
#include "paddle/fluid/memory/detail/system_allocator.h"
#include "paddle/fluid/platform/gpu_info.h"

DECLARE_double(fraction_of_gpu_memory_to_use);

namespace paddle {
namespace fluid {
namespace memory {

using BuddyAllocator = detail::BuddyAllocator;

namespace {

BuddyAllocator* GetCPUBuddyAllocator() {
  static detail::BuddyAllocator* a = nullptr;
  if (a == nullptr) {
    a = new detail::BuddyAllocator(new detail::CPUAllocator,
                                   platform::CpuMinChunkSize(),
                                   platform::CpuMaxChunkSize());
  }
  return a;
}

BuddyAllocator* GetGPUBuddyAllocator(int gpu_id) {
#ifdef PADDLE_WITH_CUDA
  static BuddyAllocator** as = NULL;
  if (as == NULL) {
    int gpu_num = platform::GetCUDADeviceCount();
    as = new BuddyAllocator*[gpu_num];
    for (int gpu = 0; gpu < gpu_num; gpu++) {
      as[gpu] = nullptr;
    }
  }
  platform::SetDeviceId(gpu_id);
  if (!as[gpu_id]) {
    as[gpu_id] = new BuddyAllocator(new detail::GPUAllocator(gpu_id),
                                    platform::GpuMinChunkSize(),
                                    platform::GpuMaxChunkSize());
    VLOG(10) << "\n\nNOTE: each GPU device use "
             << FLAGS_fraction_of_gpu_memory_to_use * 100
             << "% of GPU memory.\n"
             << "You can set GFlags environment variable '"
             << "FLAGS_fraction_of_gpu_memory_to_use"
             << "' to change the fraction of GPU usage.\n\n";
  }
  return as[gpu_id];
#else  // PADDLE_WITH_CUDA
  PADDLE_THROW("GetGPUBuddyAllocator is disabled by macro PADDLE_WITH_CUDA");
#endif  // PADDLE_WITH_CUDA
}

BuddyAllocator* GetCUDAPinnedBuddyAllocator() {
#ifdef PADDLE_WITH_CUDA
  static BuddyAllocator* ba = NULL;
  if (ba == NULL) {
    ba = new BuddyAllocator(new detail::CUDAPinnedAllocator,
                            platform::CUDAPinnedMinChunkSize(),
                            platform::CUDAPinnedMaxChunkSize());
  }
  return ba;
#else  // PADDLE_WITH_CUDA
  PADDLE_THROW("GetCUDAPinnedBuddyAllocator is disabled by macro PADDLE_WITH_CUDA");
#endif  // PADDLE_WITH_CUDA
}

// GPUID assumes that p is of type CUDAPlace and returns
// CUDAPlace.deivce.  It throws an exception if p is not CUDAPlace.
int GPUID(const platform::Place& p) {
  return dynamic_cast<const platform::CUDAPlace&>(p).device;
}

}  // namespace

void* Alloc(const platform::Place& p, size_t size) {
  VLOG(10) << "Allocate " << size << " bytes on " << p;
  void* r = platform::is_cpu_place(p) ?
            GetCPUBuddyAllocator()->Alloc(size) :
            (platform::is_gpu_place(p) ?
             GetGPUBuddyAllocator(GPUID(p))->Alloc(size) :
             (platform::is_cuda_pinned_place(p) ?
              GetCUDAPinnedBuddyAllocator()->Alloc(size) :
              nullptr));
  VLOG(10) << "  pointer=" << r;
  return r;
}

void Free(const platform::Place& p, void* m) {
  VLOG(10) << "Free pointer=" << p << " on " << p;
  if (platform::is_cpu_place(p))
    GetCPUBuddyAllocator()->Free(m);
  else if (platform::is_gpu_place(p))
    GetGPUBuddyAllocator(GPUID(p))->Free(m);
  else if (platform::is_cuda_pinned_place(p))
    GetCUDAPinnedBuddyAllocator()->Free(m);
  else
    PADDLE_THROW("Free : unknown place type");
}

size_t Used(const platform::CPUPlace& p) {
  return
      platform::is_cpu_place(p) ?
      GetCPUBuddyAllocator()->Used() :
      (platform::is_gpu_place(p) ?
       GetGPUBuddyAllocator(GPUID(p))->Used() :
       (platform::is_cuda_pinned_place(p) ?
        GetCUDAPinnedBuddyAllocator()->Used() :
        static_cast<size_t>(-1)));
}

}  // namespace memory
}  // namespace fluid
}  // namespace paddle

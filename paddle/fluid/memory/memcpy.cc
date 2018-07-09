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

#include "paddle/fluid/memory/memcpy.h"

#include <cstring>  // for memcpy

namespace paddle {
namespace fluid {
namespace memory {

template <typename DstPlace, typename SrcPlace>
void CopyImpl(const DstPlace& dst_place, void* dst,
          const SrcPlace& src_place, const void* src, size_t num);

template <>
void CopyImpl<platform::CPUPlace, platform::CPUPlace>(
    const platform::CPUPlace&,
    void* dst,
    const platform::CPUPlace&,
    const void* src,
    size_t num) {
  std::memcpy(dst, src, num);
}

#ifdef PADDLE_WITH_CUDA

template <typename DstPlace, typename SrcPlace>
void CopyImpl(const DstPlace& dst_place, void* dst,
          const SrcPlace& src_place, const void* src, size_t num,
          cudaStream_t stream);

template <>
void CopyImpl<platform::CPUPlace, platform::CUDAPlace>(
    const platform::CPUPlace& dst_place,
    void* dst,
    const platform::CUDAPlace& src_place,
    const void* src,
    size_t num,
    cudaStream_t stream) {
  platform::SetDeviceId(src_place.device);
  if (stream) {
    platform::GpuMemcpyAsync(dst, src, num, cudaMemcpyDeviceToHost, stream);
  } else {
    platform::GpuMemcpySync(dst, src, num, cudaMemcpyDeviceToHost);
  }
}

template <>
void CopyImpl<platform::CUDAPlace, platform::CPUPlace>(
    const platform::CUDAPlace& dst_place,
    void* dst,
    const platform::CPUPlace& src_place,
    const void* src,
    size_t num,
    cudaStream_t stream) {
  platform::SetDeviceId(dst_place.device);
  if (stream) {
    platform::GpuMemcpyAsync(dst, src, num, cudaMemcpyHostToDevice, stream);
  } else {
    platform::GpuMemcpySync(dst, src, num, cudaMemcpyHostToDevice);
  }
}

template <>
void CopyImpl<platform::CUDAPlace, platform::CUDAPlace>(
    const platform::CUDAPlace& dst_place,
    void* dst,
    const platform::CUDAPlace& src_place,
    const void* src,
    size_t num,
    cudaStream_t stream) {
  if (dst_place == src_place) {
    platform::SetDeviceId(src_place.device);
    if (stream) {
      platform::GpuMemcpyAsync(dst, src, num, cudaMemcpyDeviceToDevice, stream);
    } else {
      platform::GpuMemcpySync(dst, src, num, cudaMemcpyDeviceToDevice);
    }
  } else {
    if (stream) {
      platform::GpuMemcpyPeerAsync(
          dst, dst_place.device, src, src_place.device, num, stream);
    } else {
      platform::GpuMemcpyPeerSync(
          dst, dst_place.device, src, src_place.device, num);
    }
  }
}

template <>
void CopyImpl<platform::CPUPlace, platform::CUDAPinnedPlace>(
    const platform::CPUPlace& dst_place,
    void* dst,
    const platform::CUDAPinnedPlace& src_place,
    const void* src,
    size_t num,
    cudaStream_t stream) {
  std::memcpy(dst, src, num);
}

template <>
void CopyImpl<platform::CUDAPinnedPlace, platform::CPUPlace>(
    const platform::CUDAPinnedPlace& dst_place,
    void* dst,
    const platform::CPUPlace& src_place,
    const void* src,
    size_t num,
    cudaStream_t stream) {
  std::memcpy(dst, src, num);
}

template <>
void CopyImpl<platform::CUDAPinnedPlace, platform::CUDAPinnedPlace>(
    const platform::CUDAPinnedPlace& dst_place,
    void* dst,
    const platform::CUDAPinnedPlace& src_place,
    const void* src,
    size_t num,
    cudaStream_t stream) {
  std::memcpy(dst, src, num);
}

template <>
void CopyImpl<platform::CUDAPinnedPlace, platform::CUDAPlace>(
    const platform::CUDAPinnedPlace& dst_place,
    void* dst,
    const platform::CUDAPlace& src_place,
    const void* src,
    size_t num,
    cudaStream_t stream) {
  platform::SetDeviceId(src_place.device);
  if (stream) {
    platform::GpuMemcpyAsync(dst, src, num, cudaMemcpyDeviceToHost, stream);
  } else {
    platform::GpuMemcpySync(dst, src, num, cudaMemcpyDeviceToHost);
  }
}

template <>
void CopyImpl<platform::CUDAPlace, platform::CUDAPinnedPlace>(
    const platform::CUDAPlace& dst_place,
    void* dst,
    const platform::CUDAPinnedPlace& src_place,
    const void* src,
    size_t num,
    cudaStream_t stream) {
  platform::SetDeviceId(dst_place.device);
  if (stream) {
    platform::GpuMemcpyAsync(dst, src, num, cudaMemcpyHostToDevice, stream);
  } else {
    platform::GpuMemcpySync(dst, src, num, cudaMemcpyHostToDevice);
  }
}

#endif  // PADDLE_WITH_CUDA


#define MEMCPY_CASE(dst, src)               \
  (platform::is_##dst##_place(dst_place) && \
   platform::is_##src##_place(src_place))

#define MEMCPY_CALL(dstp, srcp)                                         \
  CopyImpl(dynamic_cast<const platform::dstp&>(dst_place), dst,             \
       dynamic_cast<const platform::srcp&>(src_place), src, num)

#define MEMCPY_CASE_CALL(dst, src, dstp, srcp)          \
  if MEMCPY_CASE(dst, src) MEMCPY_CALL(dstp, srcp)

void Copy(const platform::Place& dst_place, void* dst,
          const platform::Place& src_place, const void* src, size_t num) {

#define MEMCPY_CALL_NULL(dstp, srcp)                                    \
  CopyImpl(dynamic_cast<const platform::dstp&>(dst_place), dst,             \
       dynamic_cast<const platform::srcp&>(src_place), src, num, NULL)

#define MEMCPY_CASE_CALL_NULL(dst, src, dstp, srcp)          \
  if MEMCPY_CASE(dst, src) MEMCPY_CALL_NULL(dstp, srcp)
  
  MEMCPY_CASE_CALL(cpu, cpu, CPUPlace, CPUPlace);
#ifdef PADDLE_WITH_CUDA
  else MEMCPY_CASE_CALL_STREAM(cpu, gpu, CPUPlace, CUDAPlace);
  else MEMCPY_CASE_CALL_STREAM(gpu, cpu, CUDAPlace, CPUPlace);
  else MEMCPY_CASE_CALL_STREAM(gpu, gpu, CUDAPlace, CUDAPlace);
  else MEMCPY_CASE_CALL(cpu, cuda_pinned, CPUPlace, CUDAPinnedPlace);
  else MEMCPY_CASE_CALL(cuda_pinned, cpu, CUDAPinnedPlace, CPUPlace);
  else MEMCPY_CASE_CALL(cuda_pinned, cuda_pinned, CUDAPinnedPlace, CUDAPinnedPlace);
  else MEMCPY_CASE_CALL_STREAM(gpu, cuda_pinned, CUDAPlace, CUDAPinnedPlace);
  else MEMCPY_CASE_CALL_STREAM(cuda_pinned, gpu, CUDAPinnedPlace, CUDAPlace);
#endif  // PADDLE_WITH_CUDA
  
#undef MEMCPY_CALL_NULL
#undef MEMCPY_CASE_CALL_NULL
}

#ifdef PADDLE_WITH_CUDA
void Copy(const platform::Place& dst_place, void* dst,
          const platform::Place& src_place, const void* src, size_t num,
          cudaStream_t stream) {

#define MEMCPY_CALL_STREAM(dstp, srcp)                                  \
  CopyImpl(dynamic_cast<const platform::dstp&>(dst_place), dst,             \
       dynamic_cast<const platform::srcp&>(src_place), src, num, stream)

#define MEMCPY_CASE_CALL_STREAM(dst, src, dstp, srcp)          \
  if MEMCPY_CASE(dst, src) MEMCPY_CALL_STREAM(dstp, srcp)
  
  MEMCPY_CASE_CALL(cpu, cpu, CPUPlace, CPUPlace);
  else MEMCPY_CASE_CALL_STREAM(cpu, gpu, CPUPlace, CUDAPlace);
  else MEMCPY_CASE_CALL_STREAM(gpu, cpu, CUDAPlace, CPUPlace);
  else MEMCPY_CASE_CALL_STREAM(gpu, gpu, CUDAPlace, CUDAPlace);
  else MEMCPY_CASE_CALL(cpu, cuda_pinned, CPUPlace, CUDAPinnedPlace);
  else MEMCPY_CASE_CALL(cuda_pinned, cpu, CUDAPinnedPlace, CPUPlace);
  else MEMCPY_CASE_CALL(cuda_pinned, cuda_pinned, CUDAPinnedPlace, CUDAPinnedPlace);
  else MEMCPY_CASE_CALL_STREAM(gpu, cuda_pinned, CUDAPlace, CUDAPinnedPlace);
  else MEMCPY_CASE_CALL_STREAM(cuda_pinned, gpu, CUDAPinnedPlace, CUDAPlace);

#undef MEMCPY_CALL_STREAM
#undef MEMCPY_CASE_CALL_STREAM
}

#undef MEMCPY_CASE
#undef MEMCPY_CALL
#undef MEMCPY_CASE_CALL

#endif  // PADDLE_WITH_CUDA

}  // namespace memory
}  // namespace fluid
}  // namespace paddle

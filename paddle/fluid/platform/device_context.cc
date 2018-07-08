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
#include "paddle/fluid/platform/device_context.h"

#include <string>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace fluid {
namespace platform {

DeviceContextPool* DeviceContextPool::the_pool_ = nullptr;

DeviceContextPool::DeviceContextPool() {
#ifdef PADDLE_WITH_CUDA
  int num_cuda_devices = 0;
  try {
    num_cuda_devices = platform::GetCUDADeviceCount();
  } catch (const std::exception &exp) {
    LOG(WARNING) << "Compiled with WITH_GPU, but no GPU found in runtime.";
  }

  for (int i = 0; i < num_cuda_devices; ++i) {
    cuda_device_contexts_.push_back(
        std::unique_ptr<CUDADeviceContext>(
            new CUDADeviceContext(CUDAPlace(i))));
  }
#endif  // PADDLE_WITH_CUDA

  cpu_device_context_ =
      std::unique_ptr<CPUDeviceContext>(new CPUDeviceContext(CPUPlace()));
}

DeviceContext* DeviceContextPool::Get(const platform::Place& p) {
  DeviceContext* r = nullptr;
  if (is_cpu_place(p))
    r = cpu_device_context_.get();
#ifdef PADDLE_WITH_CUDA
  else if (is_gpu_place(p))
    r = cuda_device_contexts_[dynamic_cast<const CUDAPlace&>(p).device].get();
#endif  // PADDLE_WITH_CUDA
  else
    PADDLE_THROW("DeviceContextPool::Get : unknown place type");
  return r;
}

CPUDeviceContext::CPUDeviceContext() {
  eigen_device_.reset(new Eigen::DefaultDevice());
}

CPUDeviceContext::CPUDeviceContext(CPUPlace place) : place_(place) {
  eigen_device_.reset(new Eigen::DefaultDevice());
}

Eigen::DefaultDevice* CPUDeviceContext::eigen_device() const {
  return eigen_device_.get();
}

Place CPUDeviceContext::GetPlace() const { return place_; }

#ifdef PADDLE_WITH_CUDA

class EigenCudaStreamDevice : public Eigen::StreamInterface {
 public:
  EigenCudaStreamDevice() : scratch_(nullptr), semaphore_(nullptr) {
    Eigen::initializeDeviceProp();
  }
  ~EigenCudaStreamDevice() override {}

  void Reinitialize(const cudaStream_t* cuda_stream, CUDAPlace place) {
    stream_ = cuda_stream;
    place_ = place;
    device_prop_ = &Eigen::m_deviceProperties[place.device];
  }

  const cudaStream_t& stream() const override { return *stream_; }

  const cudaDeviceProp& deviceProperties() const override {
    return *device_prop_;
  }

  void* allocate(size_t num_bytes) const override {
    return paddle::fluid::memory::Alloc(place_, num_bytes);
  }

  void deallocate(void* buffer) const override {
    paddle::fluid::memory::Free(place_, buffer);
  }

  void* scratchpad() const override {
    if (scratch_ == NULL) {
      scratch_ = allocate(Eigen::kCudaScratchSize + sizeof(unsigned int));
    }
    return scratch_;
  }

  unsigned int* semaphore() const override {
    if (semaphore_ == NULL) {
      char* scratch =
          static_cast<char*>(scratchpad()) + Eigen::kCudaScratchSize;
      semaphore_ = reinterpret_cast<unsigned int*>(scratch);
      PADDLE_ENFORCE(
          cudaMemsetAsync(semaphore_, 0, sizeof(unsigned int), *stream_));
    }
    return semaphore_;
  }

 private:
  CUDAPlace place_;
  const cudaStream_t* stream_;         // not owned;
  const cudaDeviceProp* device_prop_;  // not owned;
  mutable void* scratch_;
  mutable unsigned int* semaphore_;
};

CUDADeviceContext::CUDADeviceContext(CUDAPlace place) : place_(place) {
  SetDeviceId(place_.device);
  compute_capability = GetCUDAComputeCapability(place_.device);
  multi_process = GetCUDAMultiProcessors(place_.device);
  max_threads_per_mp = GetCUDAMaxThreadsPerMultiProcessor(place_.device);
  PADDLE_ENFORCE(cudaStreamCreate(&stream_));
  eigen_stream_.reset(new EigenCudaStreamDevice());
  eigen_stream_->Reinitialize(&stream_, place);
  eigen_device_.reset(new Eigen::GpuDevice(eigen_stream_.get()));
  PADDLE_ENFORCE(dynload::cublasCreate(&cublas_handle_));
  PADDLE_ENFORCE(dynload::cublasSetStream(cublas_handle_, stream_));
  if (dynload::HasCUDNN()) {
    PADDLE_ENFORCE(dynload::cudnnCreate(&cudnn_handle_));
    PADDLE_ENFORCE(dynload::cudnnSetStream(cudnn_handle_, stream_));
  } else {
    cudnn_handle_ = nullptr;
  }
}

CUDADeviceContext::~CUDADeviceContext() {
  SetDeviceId(place_.device);
  Wait();
  PADDLE_ENFORCE(dynload::cublasDestroy(cublas_handle_));
  if (cudnn_handle_ != nullptr) {
    PADDLE_ENFORCE(dynload::cudnnDestroy(cudnn_handle_));
  }
  eigen_stream_.reset();
  eigen_device_.reset();
  PADDLE_ENFORCE(cudaStreamDestroy(stream_));
}

Place CUDADeviceContext::GetPlace() const { return place_; }

void CUDADeviceContext::Wait() const {
  PADDLE_ENFORCE(cudaStreamSynchronize(stream_));
  PADDLE_ENFORCE(cudaGetLastError());
}

int CUDADeviceContext::GetComputeCapability() const {
  return compute_capability;
}

int CUDADeviceContext::GetMaxPhysicalThreadCount() const {
  return multi_process * max_threads_per_mp;
}

Eigen::GpuDevice* CUDADeviceContext::eigen_device() const {
  return eigen_device_.get();
}

cublasHandle_t CUDADeviceContext::cublas_handle() const {
  return cublas_handle_;
}

cudnnHandle_t CUDADeviceContext::cudnn_handle() const { return cudnn_handle_; }

cudaStream_t CUDADeviceContext::stream() const { return stream_; }

CUDAPinnedDeviceContext::CUDAPinnedDeviceContext() {
  eigen_device_.reset(new Eigen::DefaultDevice());
}

CUDAPinnedDeviceContext::CUDAPinnedDeviceContext(CUDAPinnedPlace place)
    : place_(place) {
  eigen_device_.reset(new Eigen::DefaultDevice());
}

Eigen::DefaultDevice* CUDAPinnedDeviceContext::eigen_device() const {
  return eigen_device_.get();
}

Place CUDAPinnedDeviceContext::GetPlace() const { return place_; }

#endif  // PADDLE_WITH_CUDA


}  // namespace platform
}  // namespace fluid
}  // namespace paddle

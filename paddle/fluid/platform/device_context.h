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

#include <memory>
#include <unordered_map>

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/dynload/cublas.h"
#include "paddle/fluid/platform/dynload/cudnn.h"
#include "paddle/fluid/platform/gpu_info.h"
#define EIGEN_USE_GPU
#endif


#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#include "unsupported/Eigen/CXX11/Tensor"

#include "glog/logging.h"

namespace paddle {
namespace fluid {
namespace platform {

class DeviceContext {
 public:
  virtual ~DeviceContext() {}
  virtual Place GetPlace() const = 0;

  virtual void Wait() const {}
};

class CPUDeviceContext : public DeviceContext {
 public:
  CPUDeviceContext();
  explicit CPUDeviceContext(CPUPlace place);

  Eigen::DefaultDevice* eigen_device() const;

  Place GetPlace() const override;

 private:
  CPUPlace place_;
  std::unique_ptr<Eigen::DefaultDevice> eigen_device_;
};


#ifdef PADDLE_WITH_CUDA

class EigenCudaStreamDevice;

class CUDADeviceContext : public DeviceContext {
 public:
  explicit CUDADeviceContext(CUDAPlace place);
  virtual ~CUDADeviceContext();

  /*! \brief  Wait for all operations completion in the stream. */
  void Wait() const override;

  /*! \brief  Return place in the device context. */
  Place GetPlace() const override;

  /*! \brief  Return compute capability in the device context. */
  int GetComputeCapability() const;

  /*! \brief  Return the max physical thread count in the device context */
  int GetMaxPhysicalThreadCount() const;

  /*! \brief  Return eigen device in the device context. */
  Eigen::GpuDevice* eigen_device() const;

  /*! \brief  Return cublas handle in the device context. */
  cublasHandle_t cublas_handle() const;

  /*! \brief  Return cudnn  handle in the device context. */
  cudnnHandle_t cudnn_handle() const;

  /*! \brief  Return cuda stream in the device context. */
  cudaStream_t stream() const;

 private:
  CUDAPlace place_;

  std::unique_ptr<Eigen::GpuDevice> eigen_device_;
  std::unique_ptr<EigenCudaStreamDevice> eigen_stream_;

  mutable std::mutex mutex_;
  cudaStream_t stream_;
  cudnnHandle_t cudnn_handle_;
  cublasHandle_t cublas_handle_;

  int compute_capability;
  int multi_process;
  int max_threads_per_mp;
};

// Currently, CUDAPinnedDeviceContext is only used to data copying.
class CUDAPinnedDeviceContext : public DeviceContext {
 public:
  CUDAPinnedDeviceContext();
  explicit CUDAPinnedDeviceContext(CUDAPinnedPlace place);

  Place GetPlace() const override;

  Eigen::DefaultDevice* eigen_device() const;

 private:
  CUDAPinnedPlace place_;
  std::unique_ptr<Eigen::DefaultDevice> eigen_device_;
};

#endif  // PADDLE_WITH_CUDA

/*! \brief device context pool singleton */
class DeviceContextPool {
 public:
  DeviceContextPool();
  
  static DeviceContextPool& Instance() {
    if (the_pool_ == nullptr)
      the_pool_ = new DeviceContextPool();
    return *the_pool_;
  }

  DeviceContext* Get(const platform::Place& place);

  size_t size() const {
#ifdef PADDLE_WITH_CUDA
    return cuda_device_contexts_.size() + 1 /*cpu_device_context_*/;
#else  // PADDLE_WITH_CUDA
    return 1;
#endif  // PADDLE_WITH_CUDA
  }

 private:
  static DeviceContextPool* Init();
  static DeviceContextPool* the_pool_;

#ifdef PADDLE_WITH_CUDA
  std::vector<std::unique_ptr<CUDADeviceContext> > cuda_device_contexts_;
#endif  // PADDLE_WITH_CUDA
  std::unique_ptr<CPUDeviceContext> cpu_device_context_;
};

}  // namespace platform
}  // namespace fluid
}  // namespace paddle

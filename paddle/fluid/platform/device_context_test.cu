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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <vector>

TEST(Device, Init) {
  using paddle::fluid::platform::CUDADeviceContext;
  using paddle::fluid::platform::CUDAPlace;
  using paddle::fluid::platform::DeviceContext;

  int count = paddle::fluid::platform::GetCUDADeviceCount();
  for (int i = 0; i < count; i++) {
    CUDADeviceContext* device_context = new CUDADeviceContext(CUDAPlace(i));
    Eigen::GpuDevice* gpu_device = device_context->eigen_device();
    ASSERT_NE(nullptr, gpu_device);
    delete device_context;
  }
}

TEST(Device, CUDADeviceContext) {
  using paddle::fluid::platform::CUDADeviceContext;
  using paddle::fluid::platform::CUDAPlace;

  int count = paddle::fluid::platform::GetCUDADeviceCount();
  for (int i = 0; i < count; i++) {
    CUDADeviceContext* device_context = new CUDADeviceContext(CUDAPlace(i));
    Eigen::GpuDevice* gpu_device = device_context->eigen_device();
    ASSERT_NE(nullptr, gpu_device);
    cudnnHandle_t cudnn_handle = device_context->cudnn_handle();
    ASSERT_NE(nullptr, cudnn_handle);
    cublasHandle_t cublas_handle = device_context->cublas_handle();
    ASSERT_NE(nullptr, cublas_handle);
    ASSERT_NE(nullptr, device_context->stream());
    delete device_context;
  }
}

TEST(Device, DeviceContextPool) {
  using paddle::fluid::platform::CPUPlace;
  using paddle::fluid::platform::CUDADeviceContext;
  using paddle::fluid::platform::CUDAPlace;
  using paddle::fluid::platform::DeviceContextPool;
  using paddle::fluid::platform::Place;

  DeviceContextPool& pool = DeviceContextPool::Instance();
  auto cpu_dev_ctx1 = pool.Get(CPUPlace());
  auto cpu_dev_ctx2 = pool.Get(CPUPlace());
  ASSERT_EQ(cpu_dev_ctx2, cpu_dev_ctx1);

  std::vector<Place> gpu_places;
  int count = paddle::fluid::platform::GetCUDADeviceCount();
  for (int i = 0; i < count; ++i) {
    auto dev_ctx = pool.Get(CUDAPlace(i));
    ASSERT_NE(dev_ctx, nullptr);
  }
}

int main(int argc, char** argv) {
  //  std::vector<paddle::fluid::platform::Place> places;
  //
  //  places.emplace_back(paddle::fluid::platform::CPUPlace());
  //  int count = paddle::fluid::platform::GetCUDADeviceCount();
  //  for (int i = 0; i < count; ++i) {
  //    places.emplace_back(paddle::fluid::platform::CUDAPlace(i));
  //  }
  //
  //  VLOG(0) << " DeviceCount " << count;
  //  paddle::fluid::platform::DeviceContextPool::Init(places);

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

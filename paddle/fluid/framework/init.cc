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
#include <string.h>  // for strdup
#include <algorithm>
#include <stdexcept>
#include <string>

#include "paddle/fluid/framework/init.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/string/piece.h"

namespace paddle {
namespace framework {

std::once_flag gflags_init_flag;
std::once_flag p2p_init_flag;

void InitGflags(std::vector<std::string> argv) {
  std::call_once(gflags_init_flag, [&]() {
    argv.insert(argv.begin(), "dummy");
    int argc = argv.size();
    char **arr = new char *[argv.size()];
    std::string line;
    for (size_t i = 0; i < argv.size(); i++) {
      arr[i] = &argv[i][0];
      line += argv[i];
      line += ' ';
    }
    google::ParseCommandLineFlags(&argc, &arr, true);
    VLOG(1) << "Init commandline: " << line;
  });
}

void InitP2P(std::vector<int> devices) {
#ifdef PADDLE_WITH_CUDA
  std::call_once(p2p_init_flag, [&]() {
    int count = devices.size();
    for (int i = 0; i < count; ++i) {
      for (int j = 0; j < count; ++j) {
        if (devices[i] == devices[j]) continue;
        int can_acess = -1;
        PADDLE_ENFORCE(
            cudaDeviceCanAccessPeer(&can_acess, devices[i], devices[j]),
            "Failed to test P2P access.");
        if (can_acess != 1) {
          LOG(WARNING) << "Cannot enable P2P access from " << devices[i]
                       << " to " << devices[j];
        } else {
          cudaSetDevice(devices[i]);
          cudaDeviceEnablePeerAccess(devices[j], 0);
        }
      }
    }
  });
#endif
}

void InitGLOG(const std::string &prog_name) {
  // glog will not hold the ARGV[0] inside.
  // Use strdup to alloc a new string.
  google::InitGoogleLogging(strdup(prog_name.c_str()));
  google::InstallFailureSignalHandler();
}

}  // namespace framework
}  // namespace paddle
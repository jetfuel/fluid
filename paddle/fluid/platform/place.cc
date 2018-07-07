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
#include "paddle/fluid/platform/place.h"

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace fluid {
namespace platform {

namespace {
const Place* the_default_place = &default_cpu();
}  // namespace

const Place* clone_place(const Place& p) {
  Place* r = nullptr;
  if (is_cpu_place(p)) {
    r = new CPUPlace();
  } else if (is_gpu_place(p)) {
    r = new CUDAPlace(dynamic_cast<const CUDAPlace&>(p).device);
  } else if (is_cuda_pinned_place(p)) {
    r = new CUDAPinnedPlace();
  } else {
    PADDLE_THROW("clone_place: unknown place type");
  }
  return r;
}

void set_place(const Place& place) {
  if (the_default_place != nullptr) {
    delete the_default_place;
  }
  the_default_place = clone_place(place);
}

const Place& get_place() { return *the_default_place; }

const Place& default_cpu() {
  return *clone_place(CPUPlace());
}

const Place& default_gpu() {
  return *clone_place(CUDAPlace(0));
}

const Place& default_cuda_pinned() {
  return *clone_place(CUDAPinnedPlace());
}

bool is_gpu_place(const Place &p) {
  return dynamic_cast<const CUDAPlace*>(&p) != nullptr;
}

bool is_cpu_place(const Place &p) {
  return dynamic_cast<const CPUPlace*>(&p) != nullptr;
}

bool is_cuda_pinned_place(const Place &p) {
  return dynamic_cast<const CUDAPinnedPlace*>(&p) != nullptr;
}

bool places_are_same_class(const Place &p1, const Place &p2) {
  return (is_gpu_place(p1) && is_gpu_place(p2))
      || (is_cpu_place(p1) && is_cpu_place(p2))
      || (is_cuda_pinned_place(p1) && is_cuda_pinned_place(p2));
}

bool is_same_place(const Place &p1, const Place &p2) {
  return places_are_same_class(p1, p2) &&
      (is_gpu_place(p1) ?
       dynamic_cast<const CUDAPlace*>(&p1)->device ==
       dynamic_cast<const CUDAPlace*>(&p2)->device :
       true);
}

std::ostream &operator<<(std::ostream &os, const Place &p) {
  if (is_cpu_place(p)) {
    os << "CPUPlace";
  } else if (is_gpu_place(p)) {
    os << "CUDAPlace(" << dynamic_cast<const CUDAPlace&>(p).device << ")";
  } else if (is_cuda_pinned_place(p)) {
    os << "CUDAPinnedPlace";
  }
  return os;
};

int which_place(const Place& p) {
  int r = -1;
  if (is_cpu_place(p))
    r = 0;
  else if (is_gpu_place(p))
    r = 1;
  else if (is_cuda_pinned_place(p))
    r = 2;
  else
    PADDLE_THROW("PlaceHash::which : unknown place type");
  return r;
}

std::size_t PlaceHash::operator()(const Place &p) const {
  constexpr size_t num_dev_bits = 4;
  std::hash<int> ihash;
  size_t dev_id = 0;
  if (is_gpu_place(p)) {
    dev_id = dynamic_cast<const CUDAPlace&>(p).device;
  }
  return ihash(dev_id << num_dev_bits | which_place(p));
}

}  // namespace platform
}  // namespace fluid
}  // namespace paddle

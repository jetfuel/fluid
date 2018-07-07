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

#include <iostream>
#include <vector>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace fluid {
namespace platform {

struct Place {
  // We want to be able to check the real place type using the
  // predicate dynamic_cast<CPUPlace*>(p)!=nullptr.  To enable this
  // check, the base class must be a virtual class, and a virtual
  // class must (and at least need to) have a virtual destructor.
  //
  // Also, we don't want to allow users to define a variable of type
  // Place, so we make the destructuor pure virtual.  However, we have
  // a type PlaceList=vector<Place>, which prevents us from doing so.
  virtual ~Place() {}
};

struct CPUPlace : public Place {
  virtual ~CPUPlace() {}
  inline bool operator==(const CPUPlace &) const { return true; }
  inline bool operator!=(const CPUPlace &) const { return false; }
};

struct CUDAPlace : public Place {
  CUDAPlace() : CUDAPlace(0) {}
  explicit CUDAPlace(int d) : device(d) {}
  virtual ~CUDAPlace() {}
  inline bool operator==(const CUDAPlace &o) const {
    return device == o.device;
  }
  inline bool operator!=(const CUDAPlace &o) const { return device != o.device; }

  int device;
};

struct CUDAPinnedPlace : public Place {
  CUDAPinnedPlace() {}
  virtual ~CUDAPinnedPlace() {}
  inline bool operator==(const CUDAPinnedPlace &) const { return true; }
  inline bool operator!=(const CUDAPinnedPlace &) const { return false; }
};

using PlaceList = std::vector<Place>;

void set_place(const Place &);
const Place &get_place();

const Place& default_gpu();
const Place& default_cpu();
const Place& default_cuda_pinned();

bool is_gpu_place(const Place &);
bool is_cpu_place(const Place &);
bool is_cuda_pinned_place(const Place &);
bool places_are_same_class(const Place &, const Place &);
bool is_same_place(const Place &, const Place &);

int which_place(const Place& p);

struct PlaceHash {
  std::size_t operator()(const Place &p) const;
};

std::ostream &operator<<(std::ostream &, const Place &);

}  // namespace platform
}  // namespace fluid
}  // namespace paddle

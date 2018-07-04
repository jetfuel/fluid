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
#include "paddle/fluid/platform/variant.h"

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

struct CUDAPlace {
  CUDAPlace() : CUDAPlace(0) {}
  explicit CUDAPlace(int d) : device(d) {}
  virtual ~CUDAPlace() {}

  inline bool operator==(const CUDAPlace &o) const {
    return device == o.device;
  }
  inline bool operator!=(const CUDAPlace &o) const { return device != o.device; }

  int device;
};

struct CUDAPinnedPlace {
  CUDAPinnedPlace() {}
  virtual ~CUDAPinnedPlace() {}

  inline bool operator==(const CUDAPinnedPlace &) const { return true; }
  inline bool operator!=(const CUDAPinnedPlace &) const { return false; }
};

using PlaceList = std::vector<Place>;

void set_place(const Place &);
const Place &get_place();

const CUDAPlace default_gpu();
const CPUPlace default_cpu();
const CUDAPinnedPlace default_cuda_pinned();

bool is_gpu_place(const Place &);
bool is_cpu_place(const Place &);
bool is_cuda_pinned_place(const Place &);
bool places_are_same_class(const Place &, const Place &);
bool is_same_place(const Place &, const Place &);

struct PlaceHash {
  std::size_t operator()(const Place &p) const {
    constexpr size_t num_dev_bits = 4;
    std::hash<int> ihash;
    size_t dev_id = 0;
    if (is_gpu_place(p)) {
      dev_id = boost::get<CUDAPlace>(p).device;
    }
    return ihash(dev_id << num_dev_bits | p.which());
  }
};

std::ostream &operator<<(std::ostream &, const Place &);

template <typename Visitor>
struct PlaceVisitorWrapper
    : public boost::static_visitor<typename Visitor::result_type> {
  const Visitor &visitor_;
  explicit PlaceVisitorWrapper(const Visitor &visitor) : visitor_(visitor) {}

  typename Visitor::result_type operator()(const CPUPlace &cpu) const {
    return visitor_(cpu);
  }

  typename Visitor::result_type operator()(const CUDAPlace &cuda) const {
#ifdef PADDLE_WITH_CUDA
    return visitor_(cuda);
#else
    PADDLE_THROW("Paddle is not compiled with CUDA. Cannot visit cuda device");
    return typename Visitor::result_type();
#endif
  }

  typename Visitor::result_type operator()(
      const CUDAPinnedPlace &cuda_pinned) const {
#ifdef PADDLE_WITH_CUDA
    return visitor_(cuda_pinned);
#else
    PADDLE_THROW("Paddle is not compiled with CUDA. Cannot visit cuda_pinned");
    return typename Visitor::result_type();
#endif
  }
};

template <typename Visitor>
typename Visitor::result_type VisitPlace(const Place &place,
                                         const Visitor &visitor) {
  return boost::apply_visitor(PlaceVisitorWrapper<Visitor>(visitor), place);
}

}  // namespace platform
}  // namespace fluid
}  // namespace paddle

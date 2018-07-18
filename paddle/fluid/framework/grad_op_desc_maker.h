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
#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/op_desc.h"
// #include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace fluid {
namespace framework {

// GradOpDescMaker creates the gradient op for the given fwd_op. It
// adds the pairs of (gradient variable, corresponding input variable
// of fwd_op) to grad_to_var.  If an input variable of fwd_op is
// contained in no_grad_set, its gradient varialbe will be ignored or
// kEmptyVarName depending on the template argument DropEmptyIG in the
// derived classes.
class GradOpDescMaker {
 public:
  explicit GradOpDescMaker(
      const OpDesc& fwd_op,
      const std::unordered_set<std::string>& no_grad_set,
      std::unordered_map<std::string, std::string>* grad_to_var);

  virtual ~GradOpDescMaker() = default;
  virtual std::vector<std::unique_ptr<OpDesc>> operator()() const = 0;

 protected:
  std::vector<std::string> InputGrad(const std::string& name,
                                     bool drop_empty_grad = true) const;

  std::vector<std::string> OutputGrad(const std::string& name) const;

  std::vector<std::string> InputNames() const {
    return this->fwd_op_.InputNames();
  }

  std::vector<std::string> OutputNames() const {
    return this->fwd_op_.OutputNames();
  }

  std::vector<std::string> Input(const std::string& name) const {
    return fwd_op_.Input(name);
  }

  std::vector<std::string> Output(const std::string& name) const {
    return fwd_op_.Output(name);
  }

  const std::unordered_map<std::string, Attribute>& Attrs() const {
    return fwd_op_.GetAttrMap();
  }

  const Attribute& GetAttr(const std::string& name) const {
    auto& map = fwd_op_.GetAttrMap();
    auto it = map.find(name);
    PADDLE_ENFORCE(it != map.end(), "Cannot find attribute %s", name);
    return it->second;
  }

  template <typename T>
  inline const T& Attr(const std::string& name) const {
    return boost::get<T>(GetAttr(name));
  }

  std::string ForwardOpType() const { return this->fwd_op_.Type(); }

 private:
  const OpDesc& fwd_op_;
  const std::unordered_set<std::string>& no_grad_set_;
  std::unordered_map<std::string, std::string>* grad_to_var_;
};

class SingleGradOpDescMaker : public GradOpDescMaker {
 public:
  using GradOpDescMaker::GradOpDescMaker;

  std::vector<std::unique_ptr<OpDesc>> operator()() const {
    std::vector<std::unique_ptr<OpDesc>> retv;
    retv.emplace_back(this->Apply());
    return retv;
  }

 protected:
  virtual std::unique_ptr<OpDesc> Apply() const = 0;
};

template <bool DropEmptyIG = true>
class DefaultGradOpDescMaker : public SingleGradOpDescMaker {
 public:
  using SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  virtual std::unique_ptr<OpDesc> Apply() const {
    auto* grad = new OpDesc();
    grad->SetType(this->GradOpType());

    for (auto& input_param : this->InputNames()) {
      grad->SetInput(input_param, this->Input(input_param));
      grad->SetOutput(GradVarName(input_param),
                      this->InputGrad(input_param, DropEmptyIG));
    }

    for (auto& output_param : this->OutputNames()) {
      grad->SetInput(output_param, this->Output(output_param));
      grad->SetInput(GradVarName(output_param), this->OutputGrad(output_param));
    }

    grad->SetAttrMap(this->Attrs());

    return std::unique_ptr<OpDesc>(grad);
  }

  virtual std::string GradOpType() const {
    return this->ForwardOpType() + "_grad";
  }
};

class EmptyGradOpMaker : public GradOpDescMaker {
 public:
  using GradOpDescMaker::GradOpDescMaker;
  std::vector<std::unique_ptr<OpDesc>> operator()() const override {
    return {};
  }
};

}  // namespace framework
}  // namespace fluid
}  // namespace paddle

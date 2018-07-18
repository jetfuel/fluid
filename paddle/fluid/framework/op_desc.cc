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

#include "paddle/fluid/framework/op_desc.h"
#include <algorithm>
#include <functional>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>

#include "glog/logging.h"
#include "paddle/fluid/framework/op_proto_maker.h"
// #include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace fluid {
namespace framework {

template <typename T, typename RepeatedField>
inline void VectorToRepeated(const std::vector<T> &vec,
                             RepeatedField *repeated_field) {
  repeated_field->Clear();
  repeated_field->Reserve(vec.size());
  for (const auto &elem : vec) {
    *repeated_field->Add() = elem;
  }
}

OpDesc::OpDesc(const std::string &type,
               const VariableNameMap &inputs,
               const VariableNameMap &outputs,
               const AttributeMap &attrs) {
  desc_.set_type(type);
  inputs_ = inputs;
  outputs_ = outputs;
  attrs_ = attrs;
  need_update_ = true;
}

void OpDesc::CopyFrom(const OpDesc &op_desc) {
  desc_.set_type(op_desc.Type());
  inputs_ = op_desc.inputs_;
  outputs_ = op_desc.outputs_;
  attrs_ = op_desc.attrs_;
  need_update_ = true;
}

const proto::OpDesc &OpDesc::Proto() {
  Flush();
  return desc_;
}

const std::vector<std::string> &OpDesc::Input(const std::string &name) const {
  auto it = inputs_.find(name);
  PADDLE_ENFORCE(
      it != inputs_.end(), "Input %s cannot be found in Op %s", name, Type());
  return it->second;
}

std::vector<std::string> OpDesc::InputArgumentNames() const {
  std::vector<std::string> retv;
  for (auto &ipt : this->inputs_) {
    retv.insert(retv.end(), ipt.second.begin(), ipt.second.end());
  }
  return retv;
}

void OpDesc::SetInput(const std::string &param_name,
                      const std::vector<std::string> &args) {
  need_update_ = true;
  inputs_[param_name] = args;
}

const std::vector<std::string> &OpDesc::Output(const std::string &name) const {
  auto it = outputs_.find(name);
  PADDLE_ENFORCE(
      it != outputs_.end(), "Output %s cannot be found in Op %s", name, Type());
  return it->second;
}

std::vector<std::string> OpDesc::OutputArgumentNames() const {
  std::vector<std::string> retv;
  for (auto &ipt : this->outputs_) {
    retv.insert(retv.end(), ipt.second.begin(), ipt.second.end());
  }
  return retv;
}

void OpDesc::SetOutput(const std::string &param_name,
                       const std::vector<std::string> &args) {
  need_update_ = true;
  this->outputs_[param_name] = args;
}

proto::AttrType OpDesc::GetAttrType(const std::string &name) const {
  auto it = attrs_.find(name);
  PADDLE_ENFORCE(it != attrs_.end(), "Attribute %s is not found", name);
  return static_cast<proto::AttrType>(it->second.which() - 1);
}

std::vector<std::string> OpDesc::AttrNames() const {
  std::vector<std::string> retv;
  retv.reserve(attrs_.size());
  for (auto &attr : attrs_) {
    retv.push_back(attr.first);
  }
  return retv;
}

void OpDesc::SetAttr(const std::string &name, const Attribute &v) {
  this->attrs_[name] = v;
  need_update_ = true;
}

void OpDesc::SetAttrMap(
    const std::unordered_map<std::string, Attribute> &attr_map) {
  attrs_ = attr_map;
  need_update_ = true;
}

Attribute OpDesc::GetAttr(const std::string &name) const {
  auto it = attrs_.find(name);
  PADDLE_ENFORCE(it != attrs_.end(), "Attribute %s is not found", name);
  return it->second;
}

Attribute OpDesc::GetNullableAttr(const std::string &name) const {
  auto it = attrs_.find(name);
  if (it != attrs_.end()) {
    return it->second;
  } else {
    return Attribute();
  }
}

const std::unordered_map<std::string, Attribute> &OpDesc::GetAttrMap() const {
  return attrs_;
}

void OpDesc::Rename(const std::string &old_name, const std::string &new_name) {
  RenameInput(old_name, new_name);
  RenameOutput(old_name, new_name);
  need_update_ = true;
}

void OpDesc::RenameOutput(const std::string &old_name,
                          const std::string &new_name) {
  for (auto &output : outputs_) {
    std::replace(
        output.second.begin(), output.second.end(), old_name, new_name);
  }

  auto it = attrs_.find(framework::OpProtoAndCheckerMaker::OpRoleVarAttrName());
  if (it != attrs_.end()) {
    auto &op_vars = boost::get<std::vector<std::string>>(it->second);
    std::replace(op_vars.begin(), op_vars.end(), old_name, new_name);
  }

  need_update_ = true;
}

void OpDesc::RenameInput(const std::string &old_name,
                         const std::string &new_name) {
  for (auto &input : inputs_) {
    std::replace(input.second.begin(), input.second.end(), old_name, new_name);
  }

  auto it = attrs_.find(framework::OpProtoAndCheckerMaker::OpRoleVarAttrName());
  if (it != attrs_.end()) {
    auto &op_vars = boost::get<std::vector<std::string>>(it->second);
    std::replace(op_vars.begin(), op_vars.end(), old_name, new_name);
  }

  need_update_ = true;
}

struct SetAttrDescVisitor : public boost::static_visitor<void> {
  explicit SetAttrDescVisitor(proto::OpDesc::Attr *attr) : attr_(attr) {}
  mutable proto::OpDesc::Attr *attr_;
  void operator()(int v) const { attr_->set_i(v); }
  void operator()(float v) const { attr_->set_f(v); }
  void operator()(const std::string &v) const { attr_->set_s(v); }

  // Please refer to https://github.com/PaddlePaddle/Paddle/issues/7162
  template <class T,
            class = typename std::enable_if<std::is_same<bool, T>::value>::type>
  void operator()(T b) const {
    attr_->set_b(b);
  }

  void operator()(const std::vector<int> &v) const {
    VectorToRepeated(v, attr_->mutable_ints());
  }
  void operator()(const std::vector<float> &v) const {
    VectorToRepeated(v, attr_->mutable_floats());
  }
  void operator()(const std::vector<std::string> &v) const {
    VectorToRepeated(v, attr_->mutable_strings());
  }
  void operator()(const std::vector<bool> &v) const {
    VectorToRepeated(v, attr_->mutable_bools());
  }
  void operator()(int64_t v) const { attr_->set_l(v); }
  void operator()(boost::blank) const { PADDLE_THROW("Unexpected branch"); }
};

void OpDesc::Flush() {
  if (need_update_) {
    this->desc_.mutable_inputs()->Clear();
    for (auto &ipt : inputs_) {
      auto *input = desc_.add_inputs();
      input->set_parameter(ipt.first);
      VectorToRepeated(ipt.second, input->mutable_arguments());
    }

    this->desc_.mutable_outputs()->Clear();
    for (auto &opt : outputs_) {
      auto *output = desc_.add_outputs();
      output->set_parameter(opt.first);
      VectorToRepeated(opt.second, output->mutable_arguments());
    }

    this->desc_.mutable_attrs()->Clear();
    for (auto &attr : attrs_) {
      auto *attr_desc = desc_.add_attrs();
      attr_desc->set_name(attr.first);
      attr_desc->set_type(
          static_cast<proto::AttrType>(attr.second.which() - 1));
      SetAttrDescVisitor visitor(attr_desc);
      boost::apply_visitor(visitor, attr.second);
    }

    need_update_ = false;
  }
}

/* Not sure if we need the following:

static std::once_flag init_infer_shape_funcs;

static void InitInferShapeFuncs() {
  std::call_once(init_infer_shape_funcs, [] {
    auto &map = OpInfoMap::Instance();
    auto &info_map = *map.mutable_map();

    for (auto &kern_pair : OperatorWithKernel::AllOpKernels()) {
      auto op_type = kern_pair.first;
      auto &op_info = info_map.at(op_type);
      auto op = static_cast<OperatorWithKernel *>(op_info.Creator()(
          "", VariableNameMap{}, VariableNameMap{}, AttributeMap{}));
      if (op_info.infer_shape_) {  // infer_shape has been registered.
        continue;
      }
      op_info.infer_shape_ = [op](InferShapeContext *ctx) {
        op->InferShape(ctx);
      };
    }
  });
}
*/

// TODO(tony): Let us move this function to be OpInfoMap::CheckOpDesc.
/*
  void OpDesc::CheckAttrs() {
  PADDLE_ENFORCE(!Type().empty(),
                 "CheckAttr() can not be called before type is setted.");
  auto *checker = OpInfoMap::Instance().Get(Type()).Checker();
  if (checker == nullptr) {
    // checker is not configured. That operator could be generated by Paddle,
    // not by users.
    return;
  }
  checker->Check(attrs_);
}
*/

}  // namespace framework
}  // namespace fluid
}  // namespace paddle

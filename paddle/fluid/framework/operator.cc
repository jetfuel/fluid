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
#include "paddle/fluid/framework/operator.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <algorithm>

#include "paddle/fluid/framework/data_transform.h"
#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/var_type.h"

DECLARE_bool(benchmark);
DEFINE_bool(check_nan_inf,
            false,
            "Checking whether operator produce NAN/INF or not. It will be "
            "extremely slow so please use this flag wisely.");

namespace paddle {
namespace fluid {
namespace framework {

std::vector<std::tuple<platform::Place, Accelerator::Type>> kKernelPriority = {
    std::make_tuple(platform::CUDAPlace(0), Accelerator::Type::kCUDNN),
    std::make_tuple(platform::CUDAPlace(0), Accelerator::Type::kPlain),
    std::make_tuple(platform::CPUPlace(), Accelerator::Type::kMKLDNN),
    std::make_tuple(platform::CPUPlace(), Accelerator::Type::kPlain),
};

proto::VarType::Type GetDataTypeOfVar(const Variable* var) {
  if (var->IsType<framework::LoDTensor>()) {
    return framework::ToDataType(var->Get<framework::LoDTensor>().type());
  } else if (var->IsType<framework::SelectedRows>()) {
    return framework::ToDataType(
        var->Get<framework::SelectedRows>().value().type());
  } else {
    PADDLE_THROW("Var should be LoDTensor or SelectedRows");
  }
}

static DDim GetDims(const Scope& scope,
                    const std::string& name,
                    bool get_actual_dim = false) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) {
    return DDim({-1});
  }

  if (var->IsType<LoDTensor>()) {
    return var->Get<LoDTensor>().dims();
  } else if (var->IsType<SelectedRows>()) {
    if (get_actual_dim) {
      return var->Get<SelectedRows>().value().dims();
    } else {
      return var->Get<SelectedRows>().GetCompleteDims();
    }
  } else {
    return DDim({-1});
  }
}

static int GetRowSize(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) {
    return -1;
  }

  if (var->IsType<SelectedRows>()) {
    return var->Get<SelectedRows>().rows().size();
  }

  return -1;
}

static LoD GetLoD(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  auto default_lod = LoD({{}});

  if (var == nullptr) {
    return default_lod;
  }

  if (var->IsType<LoDTensor>()) {
    return var->Get<LoDTensor>().lod();
  } else {
    return default_lod;
  }
}

void OperatorBase::Run(const Scope& scope, const platform::Place& place) {
  VLOG(10) << "- " << DebugStringEx(&scope);
  if (platform::is_gpu_place(place)) {
#ifndef PADDLE_WITH_CUDA
    PADDLE_THROW("Cannot run operator on place %s", place);
#else
    auto dev_id = boost::get<platform::CUDAPlace>(place).device;
    platform::SetDeviceId(dev_id);
#endif
  }
  RunImpl(scope, place);
  VLOG(10) << "+ " << DebugStringEx(&scope);
}

bool OperatorBase::HasInputs(const std::string& name) const {
  if (inputs_.find(name) != inputs_.end()) {
    return true;
  } else {
    return false;
  }
}

std::string OperatorBase::Input(const std::string& name) const {
  auto& ins = Inputs(name);
  PADDLE_ENFORCE_LE(ins.size(),
                    1UL,
                    "Operator %s's input %s should contain only one variable.",
                    type_,
                    name);
  return ins.empty() ? kEmptyVarName : ins[0];
}

const std::vector<std::string>& OperatorBase::Inputs(
    const std::string& name) const {
  auto it = inputs_.find(name);
  PADDLE_ENFORCE(it != inputs_.end(),
                 "Operator %s does not have the input %s.",
                 type_,
                 name);
  return it->second;
}

bool OperatorBase::HasOutputs(const std::string& name) const {
  if (outputs_.find(name) != outputs_.end()) {
    return true;
  } else {
    return false;
  }
}

std::string OperatorBase::Output(const std::string& name) const {
  auto& outs = Outputs(name);
  PADDLE_ENFORCE_LE(outs.size(),
                    1UL,
                    "Operator %s's output %s should contain only one variable.",
                    type_,
                    name);
  return outs.empty() ? kEmptyVarName : outs[0];
}

const std::vector<std::string>& OperatorBase::Outputs(
    const std::string& name) const {
  auto it = outputs_.find(name);
  PADDLE_ENFORCE(it != outputs_.end(),
                 "Operator %s does not have an output called %s.",
                 type_,
                 name);
  return it->second;
}

std::string OperatorBase::DebugStringEx(const Scope* scope) const {
  std::stringstream ss;
  ss << "Op(" << type_ << "), inputs:{";
  for (auto it = inputs_.begin(); it != inputs_.end();) {
    auto& input = *it;
    ss << input.first << "[";
    for (size_t i = 0; i < input.second.size(); ++i) {
      ss << input.second[i];
      if (scope) {
        int row_size = GetRowSize(*scope, input.second[i]);
        if (row_size >= 0) {
          ss << "[row_size=" << row_size << "]";
        }
        ss << "[" << GetDims(*scope, input.second[i], true) << "]";
        ss << "(" << GetLoD(*scope, input.second[i]) << ")";
      }
      if (i != input.second.size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
    ++it;
    if (it != inputs_.end()) {
      ss << ", ";
    }
  }
  ss << "}, outputs:{";
  for (auto it = outputs_.begin(); it != outputs_.end();) {
    auto& output = *it;
    ss << output.first << "[";
    for (size_t i = 0; i < output.second.size(); ++i) {
      ss << output.second[i];
      if (scope) {
        int row_size = GetRowSize(*scope, output.second[i]);
        if (row_size >= 0) {
          ss << "[row_size=" << row_size << "]";
        }
        ss << "[" << GetDims(*scope, output.second[i], true) << "]";
        ss << "(" << GetLoD(*scope, output.second[i]) << ")";
      }
      if (i != output.second.size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
    ++it;
    if (it != outputs_.end()) {
      ss << ", ";
    }
  }
  ss << "}.";
  return ss.str();
}

OperatorBase::OperatorBase(const std::string& type,
                           const VariableNameMap& inputs,
                           const VariableNameMap& outputs,
                           const AttributeMap& attrs)
    : type_(type), inputs_(inputs), outputs_(outputs), attrs_(attrs) {
  GenerateTemporaryNames();
  // FIXME(tonyyang-svail): check this
  //  CheckAllInputOutputSet();
}

std::vector<std::string> OperatorBase::InputVars() const {
  std::vector<std::string> ret_val;
  for (auto& o : inputs_) {
    ret_val.reserve(ret_val.size() + o.second.size());
    ret_val.insert(ret_val.end(), o.second.begin(), o.second.end());
  }
  return ret_val;
}

std::vector<std::string> OperatorBase::OutputVars(bool has_intermediate) const {
  std::vector<std::string> ret_val;
  for (auto& out : outputs_) {
    for (auto& o : out.second) {
      ret_val.emplace_back(o);
    }
  }
  return ret_val;
}

// FIXME(tonyyang-svail): Operator should not depends on op_info
// void OperatorBase::CheckAllInputOutputSet() const {
//  auto& info_map = OpInfoMap::Instance();
//  auto* op_info = info_map.GetNullable(Type());
//  if (op_info == nullptr || op_info->proto_ == nullptr) return;
//
//  for (auto& in : op_info->Proto().inputs()) {
//    if (!in.dispensable()) {
//      PADDLE_ENFORCE(inputs_.find(in.name()) != inputs_.end(),
//                     "Operator %s's input, %s, is not set", Type(),
//                     in.name());
//    }
//  }
//
//  for (auto& out : op_info->Proto().outputs()) {
//    if (!out.dispensable()) {
//      PADDLE_ENFORCE(outputs_.find(out.name()) != outputs_.end(),
//                     "Operator %s's output, %s, is not set", Type(),
//                     out.name());
//    }
//  }
//}

void OperatorBase::GenerateTemporaryNames() {
  static std::atomic<size_t> gUniqId(0UL);
  for (auto& output : outputs_) {
    for (auto& output_name : output.second) {
      if (output_name == kTempVarName) {
        output_name += type_;
        output_name += "@";
        output_name += std::to_string(gUniqId.fetch_add(1));
      }
    }
  }
}

static bool VarIsTensor(const Variable* var) {
  return var->IsType<LoDTensor>() || var->IsType<SelectedRows>();
}

static const Tensor* GetTensorFromVar(Variable* var) {
  if (var->IsType<LoDTensor>()) {
    return var->GetMutable<LoDTensor>();
  } else if (var->IsType<SelectedRows>()) {
    return var->GetMutable<SelectedRows>()->mutable_value();
  } else {
    PADDLE_THROW("Variable type_id %s, expect LoDTensor/SelectedRows.",
                 var->Type().name());
  }
}

static Tensor* GetMutableTensorFromVar(Variable* var) {
  if (var->IsType<LoDTensor>()) {
    return var->GetMutable<LoDTensor>();
  } else if (var->IsType<SelectedRows>()) {
    return var->GetMutable<SelectedRows>()->mutable_value();
  } else {
    PADDLE_THROW("Variable type_id %s, expect LoDTensor/SelectedRows.",
                 var->Type().name());
  }
}

bool ExecutionContext::HasInput(const std::string& name) const {
  if (!op_.HasInputs(name)) {
    return false;
  }
  auto& ins = Inputs(name);
  size_t length = ins.size();
  if (length == 0) {
    return false;
  }
  PADDLE_ENFORCE_EQ(
      length, 1UL, "Input %s should not have more than one inputs", name);
  auto arg = ins[0];
  auto* var = arg == kEmptyVarName ? nullptr : scope_.FindVar(arg);
  return var != nullptr;
}

bool ExecutionContext::HasOutput(const std::string& name) const {
  if (!op_.HasOutputs(name)) {
    return false;
  }
  auto& outs = Outputs(name);
  size_t length = outs.size();
  if (length == 0) {
    return false;
  }
  PADDLE_ENFORCE_EQ(
      length, 1UL, "Output %s should not have more than one inputs", name);
  auto arg = outs[0];
  auto* var = arg == kEmptyVarName ? nullptr : scope_.FindVar(arg);
  return var != nullptr;
}

template <>
const Tensor* ExecutionContext::Input<Tensor>(const std::string& name) const {
  auto* var = InputVar(name);
  return var == nullptr ? nullptr
                        : GetTensorFromVar(const_cast<Variable*>(var));
}

template <>
const std::vector<const Tensor*> ExecutionContext::MultiInput<Tensor>(
    const std::string& name) const {
  auto names = op().Inputs(name);
  std::vector<const Tensor*> res;
  res.reserve(names.size());
  std::transform(names.begin(),
                 names.end(),
                 std::back_inserter(res),
                 [&](const std::string& sub_name) {
                   auto var = scope_.FindVar(sub_name);
                   return var == nullptr ? nullptr : GetTensorFromVar(var);
                 });
  return res;
}

template <>
Tensor* ExecutionContext::Output<Tensor>(const std::string& name) const {
  auto var = OutputVar(name);
  return var == nullptr ? nullptr : GetMutableTensorFromVar(var);
}

template <>
std::vector<Tensor*> ExecutionContext::MultiOutput<Tensor>(
    const std::string& name) const {
  auto names = op().Outputs(name);
  std::vector<Tensor*> res;
  res.reserve(names.size());
  std::transform(names.begin(),
                 names.end(),
                 std::back_inserter(res),
                 [&](const std::string& sub_name) {
                   auto var = scope_.FindVar(sub_name);
                   return var == nullptr ? nullptr
                                         : GetMutableTensorFromVar(var);
                 });
  return res;
}

bool OpSupportGPU(const std::string& op_type) {
  auto& all_kernels = OperatorWithKernel::AllOpKernels();
  auto it = all_kernels.find(op_type);
  if (it == all_kernels.end()) {
    // All control operator must support GPU
    return true;
  }
  for (auto& kern_pair : it->second) {
    if (platform::is_gpu_place(kern_pair.first.place_)) {
      return true;
    }
  }
  return false;
}

proto::VarType::Type RuntimeInferShapeContext::GetVarType(
    const std::string& name) const {
  auto* var = scope_.FindVar(name);
  return ToVarType(var->Type());
}

InferShapeVarPtr RuntimeInferShapeContext::GetVarPtr(const std::string& name) {
  return scope_.FindVar(name);
}

static void CheckTensorNANOrInf(const std::string& name,
                                const framework::Tensor& tensor) {
  if (tensor.memory_size() == 0) {
    return;
  }
  if (tensor.type().hash_code() != typeid(float).hash_code() &&   // NOLINT
      tensor.type().hash_code() != typeid(double).hash_code()) {  // NOLINT
    return;
  }
  PADDLE_ENFORCE(
      !framework::TensorContainsInf(tensor), "Tensor %s contains Inf", name);
  PADDLE_ENFORCE(
      !framework::TensorContainsNAN(tensor), "Tensor %s contains NAN", name);
}

void OperatorWithKernel::RunImpl(const Scope& scope,
                                 const platform::Place& place) const {
  RuntimeInferShapeContext infer_shape_ctx(*this, scope);
  this->InferShape(&infer_shape_ctx);
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(place);

  // check if op[type] has kernel registered.
  auto& all_op_kernels = AllOpKernels();
  auto kernels_iter = all_op_kernels.find(type_);
  if (kernels_iter == all_op_kernels.end()) {
    PADDLE_THROW(
        "There are no kernels which are registered in the %s operator.", type_);
  }

  ExecutionContext ctx(*this, scope, *dev_ctx);

  OpKernelMap& kernels = kernels_iter->second;

  // TODO(dzhwinter) : kernel fallback mechanism will be added when all the
  // transform functions are ready.

  // for (auto& candidate : kKernelPriority) {
  //   Do selection
  // }

  auto expected_kernel_key = this->GetExpectedKernelType(ctx);
  VLOG(3) << "expected_kernel_key:" << expected_kernel_key;

  auto kernel_iter = kernels.find(expected_kernel_key);
  if (kernel_iter == kernels.end()) {
    PADDLE_THROW("op %s does not have kernel for %s",
                 type_,
                 KernelTypeToString(expected_kernel_key));
  }

  // do data transform
  Scope& new_scope = scope.NewScope();

  std::vector<std::string> inplace_vars;
  for (auto& var_name_item : this->Inputs()) {
    for (auto& var_name : var_name_item.second) {
      auto* var = scope.FindVar(var_name);
      if (var && VarIsTensor(var)) {
        auto* tensor_in = GetTensorFromVar(var);
        if (tensor_in->IsInitialized()) {
          auto kernel_type_for_var = this->GetKernelTypeForVar(
              var_name_item.first, *tensor_in, expected_kernel_key);
          if (NeedTransform(kernel_type_for_var, expected_kernel_key)) {
            auto out_var_names = OutputVars(true);
            if (std::find(out_var_names.begin(),
                          out_var_names.end(),
                          var_name) != out_var_names.end()) {
              inplace_vars.push_back(var_name);
            }
            VLOG(3) << "Transform Variable " << var_name << " from "
                    << kernel_type_for_var << " to " << expected_kernel_key;
            auto* trans_var = new_scope.Var(var_name);
            std::shared_ptr<Tensor> out(new Tensor);
            DataTransform(expected_kernel_key,
                          kernel_type_for_var,
                          *tensor_in,
                          out.get());
            CopyVariableWithTensor(*var, *(out.get()), trans_var);
          }
        }
      }
    }
  }

  auto* new_dev_ctx = pool.Get(expected_kernel_key.place_);
  kernel_iter->second->Compute(
      ExecutionContext(*this, new_scope, *new_dev_ctx));

  for (auto& var_name : inplace_vars) {
    VLOG(3) << "share inplace var " + var_name + " back to it's original scope";
    auto* original_tensor = GetMutableTensorFromVar(scope.FindVar(var_name));
    auto* transformed_tensor = GetTensorFromVar(new_scope.FindVar(var_name));
    original_tensor->ShareDataWith(*transformed_tensor);
  }

  /*For profiling/benchmark only*/
  if (FLAGS_benchmark) {
    new_dev_ctx->Wait();
  }

  if (FLAGS_check_nan_inf) {
    for (auto& vname : OutputVars(true)) {
      auto* var = new_scope.FindVar(vname);
      if (var == nullptr) continue;
      if (var->IsType<framework::LoDTensor>()) {
        CheckTensorNANOrInf(vname, var->Get<framework::LoDTensor>());
      }
    }
  }
}

proto::VarType::Type OperatorWithKernel::IndicateDataType(
    const ExecutionContext& ctx) const {
  auto& scope = ctx.scope();
  int data_type = -1;
  for (auto& input : this->inputs_) {
    for (auto& ipt_name : input.second) {
      auto* var = scope.FindVar(ipt_name);
      if (var != nullptr) {
        const Tensor* t = nullptr;
        if (var->IsType<Tensor>()) {
          t = &var->Get<Tensor>();
        } else if (var->IsType<LoDTensor>()) {
          t = &var->Get<LoDTensor>();
        } else if (var->IsType<SelectedRows>()) {
          t = &(var->Get<SelectedRows>().value());
        }
        if (t != nullptr) {
          int tmp = static_cast<int>(ToDataType(t->type()));
          PADDLE_ENFORCE(
              tmp == data_type || data_type == -1,
              "DataType of Paddle Op %s must be the same. Get %d != %d",
              Type(),
              data_type,
              tmp);
          data_type = tmp;
        }
      }
    }
  }
  PADDLE_ENFORCE(data_type != -1, "DataType should be indicated by input");
  return static_cast<proto::VarType::Type>(data_type);
}

OpKernelType OperatorWithKernel::GetExpectedKernelType(
    const ExecutionContext& ctx) const {
  return OpKernelType(IndicateDataType(ctx), ctx.GetPlace());
}

OpKernelType OperatorWithKernel::GetKernelTypeForVar(
    const std::string& var_name,
    const Tensor& tensor,
    const OpKernelType& expected_kernel_type) const {
  return OpKernelType(
      expected_kernel_type.data_type_, tensor.place(), tensor.layout());
}

}  // namespace framework
}  // namespace fluid
}  // namespace paddle

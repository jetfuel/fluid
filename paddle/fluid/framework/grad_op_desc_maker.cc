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
#include "paddle/fluid/framework/grad_op_desc_maker.h"

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/op_desc.h"
//#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace fluid {
namespace framework {

GradOpDescMaker::GradOpDescMaker(
    const OpDesc& fwd_op,
    const std::unordered_set<std::string>& no_grad_set,
    std::unordered_map<std::string, std::string>* grad_to_var)
    : fwd_op_(fwd_op), no_grad_set_(no_grad_set), grad_to_var_(grad_to_var) {}

std::vector<std::string> GradOpDescMaker::InputGrad(
    const std::string& name, bool drop_empty_grad) const {
  std::vector<std::string> ret_val;
  auto var_names = this->Input(name);
  ret_val.reserve(var_names.size());
  std::transform(var_names.begin(),
                 var_names.end(),
                 std::back_inserter(ret_val),
                 [this](const std::string& fwd_var_name) -> std::string {
                   auto g_name = GradVarName(fwd_var_name);
                   if (no_grad_set_.count(g_name)) {
                     return kEmptyVarName;
                   } else {
                     (*this->grad_to_var_)[g_name] = fwd_var_name;
                     return g_name;
                   }
                 });
  if (!drop_empty_grad) {
    return ret_val;
  }
  PADDLE_ENFORCE_LE(var_names.size(),
                    1UL,
                    "BUG from operator developer:"
                    " for input argument with a list of variables, "
                    " drop_empty_grad is not allowed because it makes"
                    " the correspondence bewteen a variable and its gradient"
                    " ambiguous."
                    " Op type %s",
                    fwd_op_.Type());

  std::vector<std::string> dropped_ret_val;
  dropped_ret_val.reserve(ret_val.size());
  std::copy_if(ret_val.begin(),
               ret_val.end(),
               std::back_inserter(dropped_ret_val),
               [](const std::string& str) { return str != kEmptyVarName; });
  return dropped_ret_val;
}

std::vector<std::string> GradOpDescMaker::OutputGrad(
    const std::string& name) const {
  std::vector<std::string> ret_val;
  auto onames = this->Output(name);
  ret_val.reserve(onames.size());
  std::transform(onames.begin(),
                 onames.end(),
                 std::back_inserter(ret_val),
                 [this](const std::string& fwd_var_name) -> std::string {
                   auto g_name = GradVarName(fwd_var_name);
                   (*this->grad_to_var_)[g_name] = fwd_var_name;
                   return g_name;
                 });
  return ret_val;
}

}  // namespace framework
}  // namespace fluid
}  // namespace paddle

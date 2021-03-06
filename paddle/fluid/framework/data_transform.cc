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
#include "paddle/fluid/framework/data_transform.h"

#include <functional>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_device_transform.h"
#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/transform.h"

namespace paddle {
namespace fluid {
namespace framework {

namespace {

static void PassTensorData(Tensor* from, Tensor* to) {
  to->ShareDataWith(*from);
  *from = Tensor();
}

}  // namespace

void DataTransform(const OpKernelType& expected_kernel_type,
                   const OpKernelType& kernel_type_for_var,
                   const Tensor& input_tensor,
                   Tensor* output_tensor) {
  bool transformed = false;
  Tensor in;
  in.ShareDataWith(input_tensor);
  Tensor out;
  TensorDataLayout lin = kernel_type_for_var.data_layout_;
  TensorDataLayout lout = expected_kernel_type.data_layout_;

  // do layout transform
  if (NeedTransformLayout(lout, lin)) {
    TransDataLayout(kernel_type_for_var, expected_kernel_type, in, &out);
    transformed = true;
    PassTensorData(&out, &in);
  }

  // do data type transform
  if (expected_kernel_type.data_type_ != kernel_type_for_var.data_type_) {
    TransDataType(kernel_type_for_var, expected_kernel_type, in, &out);
    transformed = true;
    PassTensorData(&out, &in);
  }

  // do device transform
  if (!platform::is_same_place(kernel_type_for_var.place_,
                               expected_kernel_type.place_)) {
    TransDataDevice(in, expected_kernel_type.place_, &out);
    transformed = true;
    PassTensorData(&out, &in);
  }

  PADDLE_ENFORCE(transformed, "No transform is applied, please check!");
  // get output data
  output_tensor->ShareDataWith(in);
}

void CopyVariableWithTensor(const Variable& in_var,
                            const Tensor& tensor,
                            Variable* out_var) {
  if (in_var.IsType<LoDTensor>()) {
    auto& in_lod_tensor = in_var.Get<LoDTensor>();
    auto* tran_lod_tensor = out_var->GetMutable<LoDTensor>();
    tran_lod_tensor->set_lod(in_lod_tensor.lod());
    tran_lod_tensor->set_layout(in_lod_tensor.layout());
    tran_lod_tensor->ShareDataWith(tensor);
  } else if (in_var.IsType<SelectedRows>()) {
    auto& in_selected_rows = in_var.Get<SelectedRows>();
    auto* trans_selected_rows = out_var->GetMutable<SelectedRows>();
    trans_selected_rows->set_height(in_selected_rows.height());
    trans_selected_rows->set_rows(in_selected_rows.rows());
    trans_selected_rows->mutable_value()->ShareDataWith(tensor);
  } else {
    PADDLE_THROW("unknown var type");
  }
}

}  // namespace framework
}  // namespace fluid
}  // namespace paddle

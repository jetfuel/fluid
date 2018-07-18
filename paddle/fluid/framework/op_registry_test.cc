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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace fluid {
namespace framework {

class CosineOp : public OperatorBase {
 public:
  using OperatorBase::OperatorBase;

 private:
  void RunImpl(const Scope& scope,
               const platform::Place& place) const override {}
};

class CosineOpProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("input", "input of cosine op");
    AddOutput("output", "output of cosine op");
    AddAttr<float>("scale", "scale of cosine op")
        .SetDefault(1.0)
        .GreaterThan(0.0);
    AddComment("This is cos op");
  }
};

class MyTestOp : public OperatorBase {
 public:
  using OperatorBase::OperatorBase;

 private:
  void RunImpl(const Scope& scope,
               const platform::Place& place) const override {}
};

class MyTestOpProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("input", "input of cosine op").AsDuplicable();
    AddOutput("output", "output of cosine op").AsIntermediate();
    auto my_checker = [](int i) {
      PADDLE_ENFORCE(i % 2 == 0, "'test_attr' must be even!");
    };
    AddAttr<int>("test_attr", "a simple test attribute")
        .AddCustomChecker(my_checker);
    AddComment("This is my_test op");
  }
};
}  // namespace framework
}  // namespace fluid
}  // namespace paddle

static void BuildVar(const std::string& param_name,
                     std::initializer_list<const char*> arguments,
                     paddle::fluid::framework::proto::OpDesc::Var* var) {
  var->set_parameter(param_name);
  for (auto& arg_name : arguments) {
    var->add_arguments(arg_name);
  }
}
REGISTER_OP_WITHOUT_GRADIENT(
    cos_sim,
    paddle::fluid::framework::CosineOp,
    paddle::fluid::framework::CosineOpProtoAndCheckerMaker);
REGISTER_OP_WITHOUT_GRADIENT(
    my_test_op,
    paddle::fluid::framework::MyTestOp,
    paddle::fluid::framework::MyTestOpProtoAndCheckerMaker);

TEST(OpRegistry, CreateOp) {
  paddle::fluid::framework::proto::OpDesc op_desc;
  op_desc.set_type("cos_sim");
  BuildVar("input", {"aa"}, op_desc.add_inputs());
  BuildVar("output", {"bb"}, op_desc.add_outputs());

  float scale = 3.3;
  auto attr = op_desc.mutable_attrs()->Add();
  attr->set_name("scale");
  attr->set_type(paddle::fluid::framework::proto::AttrType::FLOAT);
  attr->set_f(scale);

  auto op = paddle::fluid::framework::OpRegistry::CreateOp(op_desc);
  paddle::fluid::framework::Scope scope;
  paddle::fluid::platform::CPUPlace cpu_place;
  op->Run(scope, cpu_place);
  float scale_get = op->Attr<float>("scale");
  ASSERT_EQ(scale_get, scale);
}

TEST(OpRegistry, IllegalAttr) {
  paddle::fluid::framework::proto::OpDesc op_desc;
  op_desc.set_type("cos_sim");
  BuildVar("input", {"aa"}, op_desc.add_inputs());
  BuildVar("output", {"bb"}, op_desc.add_outputs());

  auto attr = op_desc.mutable_attrs()->Add();
  attr->set_name("scale");
  attr->set_type(paddle::fluid::framework::proto::AttrType::FLOAT);
  attr->set_f(-2.0);

  bool caught = false;
  try {
    paddle::fluid::framework::OpRegistry::CreateOp(op_desc);
  } catch (paddle::fluid::platform::EnforceNotMet err) {
    caught = true;
    std::string msg = "larger_than check fail";
    const char* err_msg = err.what();
    for (size_t i = 0; i < msg.length(); ++i) {
      ASSERT_EQ(err_msg[i], msg[i]);
    }
  }
  ASSERT_TRUE(caught);
}

TEST(OpRegistry, DefaultValue) {
  paddle::fluid::framework::proto::OpDesc op_desc;
  op_desc.set_type("cos_sim");
  BuildVar("input", {"aa"}, op_desc.add_inputs());
  BuildVar("output", {"bb"}, op_desc.add_outputs());

  ASSERT_TRUE(op_desc.IsInitialized());

  auto op = paddle::fluid::framework::OpRegistry::CreateOp(op_desc);
  paddle::fluid::framework::Scope scope;
  paddle::fluid::platform::CPUPlace cpu_place;
  op->Run(scope, cpu_place);
  ASSERT_EQ(op->Attr<float>("scale"), 1.0);
}

TEST(OpRegistry, CustomChecker) {
  paddle::fluid::framework::proto::OpDesc op_desc;
  op_desc.set_type("my_test_op");
  BuildVar("input", {"ii"}, op_desc.add_inputs());
  BuildVar("output", {"oo"}, op_desc.add_outputs());

  // attr 'test_attr' is not set
  bool caught = false;
  try {
    paddle::fluid::framework::OpRegistry::CreateOp(op_desc);
  } catch (paddle::fluid::platform::EnforceNotMet err) {
    caught = true;
    std::string msg = "Attribute 'test_attr' is required!";
    const char* err_msg = err.what();
    for (size_t i = 0; i < msg.length(); ++i) {
      ASSERT_EQ(err_msg[i], msg[i]);
    }
  }
  ASSERT_TRUE(caught);

  // set 'test_attr' set to an illegal value
  auto attr = op_desc.mutable_attrs()->Add();
  attr->set_name("test_attr");
  attr->set_type(paddle::fluid::framework::proto::AttrType::INT);
  attr->set_i(3);
  caught = false;
  try {
    paddle::fluid::framework::OpRegistry::CreateOp(op_desc);
  } catch (paddle::fluid::platform::EnforceNotMet err) {
    caught = true;
    std::string msg = "'test_attr' must be even!";
    const char* err_msg = err.what();
    for (size_t i = 0; i < msg.length(); ++i) {
      ASSERT_EQ(err_msg[i], msg[i]);
    }
  }
  ASSERT_TRUE(caught);

  // set 'test_attr' set to a legal value
  op_desc.mutable_attrs()->Clear();
  attr = op_desc.mutable_attrs()->Add();
  attr->set_name("test_attr");
  attr->set_type(paddle::fluid::framework::proto::AttrType::INT);
  attr->set_i(4);
  auto op = paddle::fluid::framework::OpRegistry::CreateOp(op_desc);
  paddle::fluid::platform::CPUPlace cpu_place;
  paddle::fluid::framework::Scope scope;
  op->Run(scope, cpu_place);
  int test_attr = op->Attr<int>("test_attr");
  ASSERT_EQ(test_attr, 4);
}

class CosineOpComplete : public paddle::fluid::framework::CosineOp {
 public:
  DEFINE_OP_CONSTRUCTOR(CosineOpComplete, paddle::fluid::framework::CosineOp);
  DEFINE_OP_CLONE_METHOD(CosineOpComplete);
};

TEST(OperatorRegistrar, Test) {
  paddle::fluid::framework::OperatorRegistrar<
      CosineOpComplete,
      paddle::fluid::framework::CosineOpProtoAndCheckerMaker>
      reg("cos");
}

namespace paddle {
namespace fluid {
namespace framework {

class OpKernelTestMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() { AddComment("NoGradOp, same input output. no Grad"); }
};

class OpWithKernelTest : public OperatorWithKernel {
 public:
  using OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(InferShapeContext* ctx) const override {}

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(proto::VarType::FP32, ctx.device_context().GetPlace());
  }
};

template <typename DeviceContext, typename T>
class OpKernelTest : public paddle::fluid::framework::OpKernel<T> {
 public:
  void Compute(const paddle::fluid::framework::ExecutionContext& ctx) const {}
};

}  // namespace framework
}  // namespace fluid
}  // namespace paddle

REGISTER_OP_WITHOUT_GRADIENT(op_with_kernel,
                             paddle::fluid::framework::OpWithKernelTest,
                             paddle::fluid::framework::OpKernelTestMaker);
REGISTER_OP_CPU_KERNEL(op_with_kernel,
                       paddle::fluid::framework::OpKernelTest<
                           paddle::fluid::platform::CPUDeviceContext,
                           float>);

REGISTER_OP_CUDA_KERNEL(op_with_kernel,
                        paddle::fluid::framework::OpKernelTest<
                            paddle::fluid::platform::CUDADeviceContext,
                            float>);

TEST(OperatorRegistrar, CPU) {
  paddle::fluid::framework::proto::OpDesc op_desc;
  paddle::fluid::platform::CPUPlace cpu_place;
  paddle::fluid::framework::Scope scope;

  op_desc.set_type("op_with_kernel");
  auto op = paddle::fluid::framework::OpRegistry::CreateOp(op_desc);

  op->Run(scope, cpu_place);
}

TEST(OperatorRegistrar, CUDA) {
  paddle::fluid::framework::proto::OpDesc op_desc;
  paddle::fluid::platform::CUDAPlace cuda_place(0);
  paddle::fluid::framework::Scope scope;

  op_desc.set_type("op_with_kernel");
  auto op = paddle::fluid::framework::OpRegistry::CreateOp(op_desc);

  op->Run(scope, cuda_place);
}

static int op_test_value = 0;

using paddle::fluid::platform::CPUDeviceContext;
using paddle::fluid::platform::CUDADeviceContext;
using paddle::fluid::platform::DeviceContext;

namespace paddle {
namespace fluid {
namespace framework {

class OpWithMultiKernelTest : public OperatorWithKernel {
 public:
  using OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(InferShapeContext* ctx) const override {}

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(proto::VarType::FP32,
                                   platform::CUDAPlace(0),
                                   TensorDataLayout::kAnyLayout,
                                   framework::Accelerator::kCUDNN);
  }
};

template <typename DeviceContext, typename T>
class OpMultiKernelTest : public paddle::fluid::framework::OpKernel<T> {
 public:
  void Compute(const paddle::fluid::framework::ExecutionContext& ctx) const;
};

template <typename T>
class OpMultiKernelTest<CPUDeviceContext, T>
    : public paddle::fluid::framework::OpKernel<T> {
 public:
  void Compute(const paddle::fluid::framework::ExecutionContext& ctx) const {
    ++op_test_value;
  }
};

template <typename T>
class OpMultiKernelTest<CUDADeviceContext, T>
    : public paddle::fluid::framework::OpKernel<T> {
 public:
  void Compute(const paddle::fluid::framework::ExecutionContext& ctx) const {
    --op_test_value;
  }
};

template <typename DeviceContext, typename T>
class OpMultiKernelTest2 : public paddle::fluid::framework::OpKernel<T> {
 public:
  void Compute(const paddle::fluid::framework::ExecutionContext& ctx) const;
};

template <typename T>
class OpMultiKernelTest2<CPUDeviceContext, T>
    : public paddle::fluid::framework::OpKernel<T> {
 public:
  void Compute(const paddle::fluid::framework::ExecutionContext& ctx) const {
    op_test_value += 10;
  }
};

template <typename T>
class OpMultiKernelTest2<CUDADeviceContext, T>
    : public paddle::fluid::framework::OpKernel<T> {
 public:
  void Compute(const paddle::fluid::framework::ExecutionContext& ctx) const {
    op_test_value -= 10;
  }
};

}  // namespace framework
}  // namespace fluid
}  // namespace paddle

REGISTER_OP_WITHOUT_GRADIENT(op_with_multi_kernel,
                             paddle::fluid::framework::OpWithMultiKernelTest,
                             paddle::fluid::framework::OpKernelTestMaker);
REGISTER_OP_KERNEL(
    op_with_multi_kernel,
    CPU,
    paddle::fluid::platform::CPUPlace,
    paddle::fluid::framework::OpMultiKernelTest<CPUDeviceContext, float>);
REGISTER_OP_KERNEL(
    op_with_multi_kernel,
    MKLDNN,
    paddle::fluid::platform::CPUPlace,
    paddle::fluid::framework::OpMultiKernelTest2<CPUDeviceContext, float>);
REGISTER_OP_KERNEL(
    op_with_multi_kernel,
    CUDA,
    paddle::fluid::platform::CUDAPlace,
    paddle::fluid::framework::OpMultiKernelTest<CUDADeviceContext, float>);
REGISTER_OP_KERNEL(
    op_with_multi_kernel,
    CUDNN,
    paddle::fluid::platform::CUDAPlace,
    paddle::fluid::framework::OpMultiKernelTest2<CUDADeviceContext, float>);

TEST(OperatorRegistrar, OpWithMultiKernel) {
  paddle::fluid::framework::proto::OpDesc op_desc;
  paddle::fluid::platform::CUDAPlace cuda_place(0);
  paddle::fluid::platform::CPUPlace cpu_place;
  paddle::fluid::framework::Scope scope;

  op_desc.set_type("op_with_multi_kernel");
  auto op = paddle::fluid::framework::OpRegistry::CreateOp(op_desc);

  // TODO(qiao) add priority back
  // use all available kernels
  op->Run(scope, cuda_place);
  EXPECT_EQ(op_test_value, -10);
}

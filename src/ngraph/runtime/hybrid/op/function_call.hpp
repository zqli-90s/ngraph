//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace runtime
    {
        class HostTensor;
        namespace hybrid
        {
            namespace op
            {
                class FunctionCall;
            }
        }
    }
}

class ngraph::runtime::hybrid::op::FunctionCall : public ngraph::op::Op
{
public:
    FunctionCall(const NodeVector& arguments,
                 const std::vector<std::pair<element::Type, Shape>>& output_info,
                 std::shared_ptr<Function> function = nullptr,
                 const std::string& backend = "");

    void execute(const std::vector<std::shared_ptr<HostTensor>>& outputs,
                 const std::vector<std::shared_ptr<HostTensor>>& inputs);

private:
    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

    const std::vector<std::pair<element::Type, Shape>> m_output_info;
    std::shared_ptr<Function> m_function;
    const std::string& m_backend;
};

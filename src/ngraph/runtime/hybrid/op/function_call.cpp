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

#include "function_call.hpp"

using namespace std;
using namespace ngraph;

runtime::hybrid::op::FunctionCall::FunctionCall(
    const NodeVector& arguments,
    const vector<pair<element::Type, Shape>>& output_info,
    shared_ptr<Function> function,
    const string& backend)
    : Op("HybridFunction", arguments)
{
    set_output_size(output_info.size());
    for (size_t i = 0; i < output_info.size(); i++)
    {
        set_output_type(i, output_info[i].first, output_info[i].second);
    }
}

shared_ptr<Node>
    runtime::hybrid::op::FunctionCall::copy_with_new_args(const NodeVector& new_args) const
{
    return nullptr;
}

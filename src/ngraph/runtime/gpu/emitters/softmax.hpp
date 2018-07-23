/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include "ngraph/op/softmax.hpp"
#include "ngraph/runtime/gpu/emitter.hpp"
#include "ngraph/runtime/gpu/op/memory_wrapped_node.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            template <>
            class Emitter<op::Softmax>
            {
            public:
                Emitter<op::Softmax>(Node* node)
                    : m_node(static_cast<op::gpu::MemoryWrappedNode<op::Softmax>*>(node))
                {
                }

                // Retrieve shapes of workspace requirements
                std::vector<Shape> get_workspaces();

                // Retrieve constant data needed for kernel execution
                std::vector<std::vector<int>> get_constants();

            public:
                // static op emitter
                void emit(GPU_ExternalFunction* external_function,
                          codegen::CodeWriter& writer,
                          const std::vector<GPU_TensorViewWrapper>& args,
                          const std::vector<GPU_TensorViewWrapper>& out);

            private:
                op::gpu::MemoryWrappedNode<op::Softmax>* m_node;
            };
        }
    }
}

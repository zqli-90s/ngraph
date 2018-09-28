//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include <memory> // std::shared_ptr
#include <vector> // std::vector

#include <onnx.hpp>
#include <onnxifi.h>

#include "ngraph/function.hpp"
#include "ngraph/runtime/tensor_view.hpp"

#include "backend.hpp"
#include "span.hpp"
#include "tensor.hpp"
#include "weights.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        class Backend;

        /// \brief Representation of onnxGraph
        class Graph
        {
        public:
            Graph(const Graph&) = delete;
            Graph& operator=(const Graph&) = delete;

            Graph() = delete;

            Graph(Graph&&) noexcept = default;
            Graph& operator=(Graph&&) noexcept = delete;

            explicit Graph(const Backend& backend)
                : m_backend{backend}
            {
            }

            void load(std::istream& sin, const Span<::onnxTensorDescriptorV1>& weights);

            void set_inputs(const Span<::onnxTensorDescriptorV1>& inputs);
            void set_outputs(const Span<::onnxTensorDescriptorV1>& outputs);

            bool compile();

            void configure_memory_fences(const ::onnxMemoryFenceV1* input_fence,
                                         ::onnxMemoryFenceV1* output_fence);

            bool operator==(const Graph& other) const noexcept;
            bool operator!=(const Graph& other) const noexcept;

            bool run_graph();

            void from_ng_outputs(const std::vector<std::shared_ptr<runtime::TensorView>>& ng_outputs,
                                 std::vector<OutputTensor>& output) const
            {
                for (std::size_t i{0}; i < ng_outputs.size(); ++i)
                {
                    output[i].from_ng(*ng_outputs[i]);
                }
            }

        private:
            std::shared_ptr<Function> m_function{nullptr};
            std::vector<std::shared_ptr<runtime::TensorView>> m_ng_inputs{};
            std::vector<OutputTensor> m_outputs{};
            std::vector<std::shared_ptr<runtime::TensorView>> m_ng_outputs{};
            const Backend& m_backend;
            const ::onnxMemoryFenceV1* m_input_fence{nullptr};
            ::onnxMemoryFenceV1* m_output_fence{nullptr};

            // void validate_tensor_descriptors(const Span<::onnxTensorDescriptorV1>& descriptors) const;
        };

        inline bool Graph::operator==(const Graph& other) const noexcept
        {
            return (m_function == other.m_function);
        }

        inline bool Graph::operator!=(const Graph& other) const noexcept
        {
            return !(*this == other);
        }

    } // namespace onnxifi

} // namespace ngraph

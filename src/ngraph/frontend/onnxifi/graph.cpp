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

#include <onnxifi.h>

#include "backend.hpp"
#include "graph.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        bool Graph::run_graph()
        {
            ::onnxStatus status{::onnxWaitEvent(m_input_fence->event)};
            if (status != ONNXIFI_STATUS_SUCCESS)
            {
                throw status::runtime{status};
            }
            bool result{m_backend.call(m_function, m_inputs, m_outputs)};
            status = ::onnxSignalEvent(m_output_fence->event);
            if (status != ONNXIFI_STATUS_SUCCESS)
            {
                throw status::runtime{status};
            }
            return result;
        }

        void Graph::configure_memory_fences(const ::onnxMemoryFenceV1* input_fence,
                                            ::onnxMemoryFenceV1* output_fence)
        {
            if ((input_fence == nullptr) || (output_fence == nullptr))
            {
                throw status::null_pointer{};
            }
            if ((input_fence->tag != ONNXIFI_TAG_MEMORY_FENCE_V1) ||
                (output_fence->tag != ONNXIFI_TAG_MEMORY_FENCE_V1))
            {
                throw status::unsupported_tag{};
            }
            if ((input_fence->type == ONNXIFI_SYNCHRONIZATION_IMPLICIT) ||
                (output_fence->type == ONNXIFI_SYNCHRONIZATION_IMPLICIT))
            {
                throw status::unsupported_fence_type{};
            }
            if ((input_fence->type != ONNXIFI_SYNCHRONIZATION_EVENT) ||
                (output_fence->type != ONNXIFI_SYNCHRONIZATION_EVENT))
            {
                throw status::invalid_fence_type{};
            }
            ::onnxEventState state;
            ::onnxStatus status{::onnxGetEventState(output_fence->event, &state)};
            if (status == ONNXIFI_STATUS_INVALID_EVENT)
            {
                status = ::onnxInitEvent(m_backend.get_handle(), &output_fence->event);
                if (status != ONNXIFI_STATUS_SUCCESS)
                {
                    throw status::runtime{status};
                }
                status = ::onnxGetEventState(output_fence->event, &state);
            }
            if (status != ONNXIFI_STATUS_SUCCESS)
            {
                throw status::runtime{status};
            }
            if (state != ONNXIFI_EVENT_STATE_NONSIGNALLED)
            {
                throw status::invalid_state{};
            }
            status = ::onnxGetEventState(input_fence->event, &state);
            if (status != ONNXIFI_STATUS_SUCCESS)
            {
                throw status::runtime{status};
            }
            if (state != ONNXIFI_EVENT_STATE_NONSIGNALLED)
            {
                throw status::invalid_state{};
            }
            m_input_fence = input_fence;
            m_output_fence = output_fence;
        }

        bool Graph::compile() { return m_backend.compile(m_function); }
        void Graph::set_weights(const Span<onnxTensorDescriptorV1>& weights)
        {
            if (weights.data() != nullptr)
            {
                if (weights.empty())
                {
                    throw status::invalid_size{};
                }

                /* TODO: apply weights to the graph */
            }
            else
            {
                if (!weights.empty())
                {
                    throw status::null_pointer{};
                }
            }
        }

        /* void Graph::validate_tensor_descriptors(const Span<::onnxTensorDescriptorV1>& descriptors) const
        {
            for (const auto& descriptor : descriptors)
            {
                if (descriptor.tag != ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1)
                {
                    throw status::unsupported_tag{};
                }
                if (descriptor.name == nullptr)
                {
                    throw status::invalid_name{};
                }
                switch (descriptor.dataType)
                {
                    case ONNXIFI_DATATYPE_FLOAT16:
                    case ONNXIFI_DATATYPE_FLOAT32:
                    case ONNXIFI_DATATYPE_FLOAT64:
                    case ONNXIFI_DATATYPE_INT8:
                    case ONNXIFI_DATATYPE_INT16:
                    case ONNXIFI_DATATYPE_INT32:
                    case ONNXIFI_DATATYPE_INT64:
                    case ONNXIFI_DATATYPE_UINT8:
                    case ONNXIFI_DATATYPE_UINT16:
                    case ONNXIFI_DATATYPE_UINT32:
                    case ONNXIFI_DATATYPE_UINT64:
                        break;
                    case ONNXIFI_DATATYPE_COMPLEX64:
                    case ONNXIFI_DATATYPE_COMPLEX128:
                        throw status::invalid_datatype{};
                    default:
                        throw status::unsupported_datatype{};
                }
                switch (descriptor.memoryType)
                {
                    case ONNXIFI_MEMORY_TYPE_CPU:
                        break;
                    case ONNXIFI_MEMORY_TYPE_CUDA_BUFFER:
                    case ONNXIFI_MEMORY_TYPE_OPENCL_BUFFER:
                    case ONNXIFI_MEMORY_TYPE_OPENGLES_TEXTURE_2D:
                    case ONNXIFI_MEMORY_TYPE_D3D_RESOURCE:
                        throw status::invalid_memory_type{};
                    default:
                        throw status::unsupported_memory_type{};
                }
                if ((descriptor.dimensions != 0) &&
                    (descriptor.shape == nullptr))
                {
                    throw status::null_pointer{};
                }
                if (descriptor.shape != nullptr)
                {
                    Span<uint64_t> shape{descriptor.shape, descriptor.dimensions};
                    for (const auto& value : shape)
                    {
                        if (value == 0)
                        {
                            throw status::invalid_shape{};
                        }
                    }
                }
                if (descriptor.buffer == 0)
                {
                    throw status::invalid_memory_location{};
                }
            }
        } */

    } // namespace onnxifi

} // namespace ngraph

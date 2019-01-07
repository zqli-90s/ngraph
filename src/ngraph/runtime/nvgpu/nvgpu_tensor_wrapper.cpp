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
#include <limits>

#include "ngraph/descriptor/layout/tensor_layout.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/runtime/nvgpu/nvgpu_tensor_wrapper.hpp"

using namespace std;
using namespace ngraph;

runtime::nvgpu::NVTensorWrapper::NVTensorWrapper(const shared_ptr<descriptor::Tensor>& tv,
                                                 const string& alias)
    : m_tensor(tv)
    , m_alias(alias)
    , m_offset(std::make_pair(runtime::nvgpu::NVTensorWrapper::TensorType::UNKNOWN,
                              std::numeric_limits<size_t>::max()))
{
}

runtime::nvgpu::NVTensorWrapper::NVTensorWrapper(const std::shared_ptr<descriptor::Tensor>& tv,
                                                 runtime::nvgpu::NVTensorWrapper::TensorType type,
                                                 size_t offset,
                                                 const std::string& alias)
    : m_tensor(tv)
    , m_alias(alias)
    , m_offset(std::make_pair(type, offset))
{
}

size_t runtime::nvgpu::NVTensorWrapper::get_size() const
{
    return m_tensor->get_tensor_layout()->get_size();
}

const Shape& runtime::nvgpu::NVTensorWrapper::get_shape() const
{
    return m_tensor->get_tensor_layout()->get_shape();
}

Strides runtime::nvgpu::NVTensorWrapper::get_strides() const
{
    return m_tensor->get_tensor_layout()->get_strides();
}

const element::Type& runtime::nvgpu::NVTensorWrapper::get_element_type() const
{
    return m_tensor->get_tensor_layout()->get_element_type();
}

const std::string& runtime::nvgpu::NVTensorWrapper::get_name() const
{
    if (m_alias.empty())
    {
        return m_tensor->get_name();
    }
    else
    {
        return m_alias;
    }
}

const std::pair<runtime::nvgpu::NVTensorWrapper::TensorType, size_t>&
    runtime::nvgpu::NVTensorWrapper::get_offset() const
{
    return m_offset;
}

const std::string& runtime::nvgpu::NVTensorWrapper::get_type() const
{
    return get_element_type().c_type_string();
}

std::ostream& ngraph::runtime::nvgpu::operator<<(std::ostream& out,
                                               const ngraph::runtime::nvgpu::NVTensorWrapper& obj)
{
    static std::vector<std::string> types{"CONSTANT", "INTERMEDIATE", "INPUT", "OUTPUT", "UNKNOWN"};
    out << "nvgpu::tensor { name: " << obj.m_tensor->get_name()
        << " tensor_type: " << types.at(static_cast<size_t>(obj.m_offset.first))
        << ", offset/index: " << obj.m_offset.second << ", dtype: " << obj.get_element_type()
        << ", shape: " << obj.get_shape() << ", size: " << obj.get_size()
        << ", alias: " << obj.m_alias << " }";
    return out;
}
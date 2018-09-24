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

#include <mutex>

#include "event.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        Event::Event(Event&& other) noexcept
        {
            std::lock_guard<std::mutex> lock{other.m_mutex};
            m_signaled = other.m_signaled;
        }

        Event& Event::operator=(Event&& other) noexcept
        {
            if (this != &other)
            {
                std::unique_lock <std::mutex> lock{m_mutex, std::defer_lock};
                std::unique_lock <std::mutex> other_lock{other.m_mutex, std::defer_lock};
                m_signaled = other.m_signaled;
            }
            return *this;
        }

    } // namespace onnxifi

} // namespace ngraph

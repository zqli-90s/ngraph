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

#include <mutex>
#include <condition_variable>

namespace ngraph
{
    namespace onnxifi
    {
        /// \brief Implementation of onnxEvent data type.
        class Event
        {
        public:
            Event(const Event&) = delete;
            Event& operator=(const Event&) = delete;

            Event(Event&&) noexcept;
            Event& operator=(Event&&) noexcept;

            Event() = default;

            void signal()
            {
                std::lock_guard<std::mutex> lock{m_mutex};
                m_signaled = true;
                m_condition_variable.notify_all();
            }

            void reset()
            {
                std::lock_guard<std::mutex> lock{m_mutex};
                m_signaled = false;
            }

            void wait() const
            {
                std::unique_lock<std::mutex> lock{m_mutex};
                m_condition_variable.wait(lock, [&]{
                    return m_signaled;
                });
            }

            template <typename Rep, typename Period>
            bool wait_for(const std::chrono::duration<Rep,Period>& duration) const
            {
                std::unique_lock<std::mutex> lock{m_mutex};
                return m_condition_variable.wait_for(lock, duration, [&]{
                    return m_signaled;
                }) == std::cv_status::no_timeout;
            }

            template <typename Clock, typename Duration>
            bool wait_until(const std::chrono::time_point<Clock,Duration>& time_point) const
            {
                std::unique_lock<std::mutex> lock{m_mutex};
                return m_condition_variable.wait_until(lock, time_point, [&]{
                    return m_signaled;
                }) == std::cv_status::no_timeout;
            }

            bool is_signaled() const
            {
                std::lock_guard<std::mutex> lock{m_mutex};
                return m_signaled;
            }

        private:
            mutable std::mutex m_mutex{};
            mutable std::condition_variable m_condition_variable{};
            bool m_signaled{false};
        };

    } // namespace onnxifi

} // namespace ngraph

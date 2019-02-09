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

#include <memory>
#include <unordered_map>
#include <vector>

#include "ngraph/function.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/result.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace hybrid
        {
            // Split function to function(s) with unique placement
            std::pair<
                std::vector<std::shared_ptr<Function>>,
                std::unordered_map<std::shared_ptr<op::Parameter>, std::shared_ptr<op::Result>>>
                split_function_by_placement(const std::shared_ptr<Function>& f);

            class Edge;
        }
    }
}

class ngraph::runtime::hybrid::Edge
{
public:
    Edge(std::shared_ptr<Node> source,
         size_t source_output_index,
         std::shared_ptr<Node> target,
         size_t target_input_index);

    static std::vector<Edge> from(std::shared_ptr<Node> source, std::shared_ptr<Node> target);

    std::shared_ptr<Node> get_source() const;
    size_t get_source_output_index() const;
    std::shared_ptr<Node> get_target() const;
    size_t get_target_input_index() const;

private:
    std::shared_ptr<Node> m_source;
    size_t m_source_output_index;
    std::shared_ptr<Node> m_target;
    size_t m_target_input_index;
};

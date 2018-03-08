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

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ngraph/log.hpp"

namespace ngraph
{
    class Function;
    class Node;

    namespace op
    {
        class Parameter;
    }

    enum class Placement
    {
        DEFAULT,
        INTERPRETER,
        CPU,
        GPU,
        ARGON,
    };

    class Cluster
    {
    public:
        Cluster();
        Cluster(const std::unordered_set<std::shared_ptr<Node>>& nodes);
        Placement get_placement() const { return m_placement; }
        std::string get_name() const;
        // Nodes
        void insert_node(const std::shared_ptr<Node>& node);
        const std::unordered_set<std::shared_ptr<Node>>& get_nodes() const { return m_nodes; }
        void clear_nodes() { m_nodes.clear(); }
        size_t size() const { return m_nodes.size(); }
        // Children
        const std::unordered_set<std::shared_ptr<Cluster>>& get_children() const
        {
            return m_children;
        }
        void insert_child(const std::shared_ptr<Cluster>& cluster);
        void remove_child(const std::shared_ptr<Cluster>& cluster);
        void remove_child_if_exists(const std::shared_ptr<Cluster>& cluster);
        bool exist_child(const std::shared_ptr<Cluster>& cluster) const;
        void clear_children() { m_children.clear(); }
        // Parents
        const std::unordered_set<std::shared_ptr<Cluster>>& get_parents() const
        {
            return m_parents;
        }
        void insert_parent(const std::shared_ptr<Cluster>& cluster);
        void remove_parent(const std::shared_ptr<Cluster>& cluster);
        void remove_parent_if_exists(const std::shared_ptr<Cluster>& cluster);
        bool exist_parent(const std::shared_ptr<Cluster>& cluster) const;
        void clear_parents() { m_parents.clear(); }
        // Debugging support
        std::string get_debug_node_names() const;

    protected:
        size_t m_instance_id;
        static std::atomic<size_t> m_next_instance_id;
        std::unordered_set<std::shared_ptr<Node>> m_nodes;
        std::unordered_set<std::shared_ptr<Cluster>> m_children;
        std::unordered_set<std::shared_ptr<Cluster>> m_parents;
        Placement m_placement = Placement::DEFAULT;
    };

    namespace cluster_util
    {
        // Cluster graph utils
        std::vector<std::shared_ptr<Cluster>>
            split_function_to_clusters(const std::shared_ptr<Function>& f);
        std::shared_ptr<Cluster> merge_clusters(const std::shared_ptr<Cluster>& src_cluster,
                                                const std::shared_ptr<Cluster>& dst_cluster);
        bool is_reachable_from_children_ptrs(std::shared_ptr<Cluster> src_cluster,
                                             std::shared_ptr<Cluster> dst_cluster);
        bool is_edge_contractable(const std::shared_ptr<Cluster>& src_cluster,
                                  const std::shared_ptr<Cluster>& dst_cluster);
        std::vector<std::shared_ptr<Cluster>>
            build_singleton_clusters(const std::shared_ptr<Function>& f);

        // Cluster passes
        void merge_adjacent_clusters_pass(std::vector<std::shared_ptr<Cluster>>& clusters);
        void merge_disjoint_clusters_pass(std::vector<std::shared_ptr<Cluster>>& clusters);
        void topological_sort_clusters_pass(std::vector<std::shared_ptr<Cluster>>& clusters);
        void node_consistency_check_pass(const std::vector<std::shared_ptr<Cluster>>& clusters,
                                         const std::shared_ptr<Function>& f);
    }

    std::string placement_to_string(Placement placement);
}

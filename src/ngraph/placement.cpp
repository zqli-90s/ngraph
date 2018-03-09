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

#include "ngraph/placement.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/util.hpp"

#include <deque>
#include <sstream>

using namespace std;
using namespace ngraph;

std::string ngraph::placement_to_string(Placement placement)
{
    // Provides safety and avoid clang's warning
    if (placement != Placement::DEFAULT && placement != Placement::INTERPRETER &&
        placement != Placement::CPU && placement != Placement::GPU && placement != Placement::ARGON)
    {
        throw ngraph_error("Placement is uninitialized.");
    }
    switch (placement)
    {
    case Placement::DEFAULT: return "DEFAULT";
    case Placement::INTERPRETER: return "INTERPRETER";
    case Placement::CPU: return "CPU";
    case Placement::GPU: return "GPU";
    case Placement::ARGON: return "ARGON";
    }
}

atomic<size_t> Cluster::m_next_instance_id(0);

Cluster::Cluster()
    : m_instance_id(m_next_instance_id.fetch_add(1))
{
}

Cluster::Cluster(const unordered_set<shared_ptr<Node>>& nodes)
    : m_instance_id(m_next_instance_id.fetch_add(1))
{
    // Check placement consistency by calling insert_node()
    for (auto node : nodes)
    {
        insert_node(node);
    }
}

string Cluster::get_name() const
{
    return "Cluster_" + to_string(m_instance_id);
}

string Cluster::get_debug_node_names() const
{
    vector<string> node_names;
    for (auto node : m_nodes)
    {
        node_names.push_back(node->get_name());
    }

    std::stringstream ss;
    ss << "[" << get_name() << "(" << placement_to_string(m_placement) << ")]{";
    ss << join(node_names);
    ss << "}";
    return ss.str();
}

void Cluster::insert_node(const std::shared_ptr<Node>& node)
{
    Placement node_placement = node->get_placement();
    if (node_placement == Placement::DEFAULT)
    {
        throw ngraph_error("Node " + node->get_name() + "has DEFAULT placement." +
                           "A Node must have a device placement to be added to a Cluster.");
    }
    if (this->size() == 0)
    {
        m_placement = node_placement;
    }
    else if (m_placement != node_placement)
    {
        throw ngraph_error("Node's placement different from cluster's placement");
    }
    m_nodes.insert(node);
}

void Cluster::insert_child(const shared_ptr<Cluster>& child)
{
    m_children.insert(child);
}

void Cluster::remove_child(const shared_ptr<Cluster>& child)
{
    if (m_children.find(child) == m_children.end())
    {
        throw ngraph_error("The child cluster to remove is not a child.");
    }
    m_children.erase(child);
}

void Cluster::remove_child_if_exists(const shared_ptr<Cluster>& child)
{
    m_children.erase(child);
}

bool Cluster::exist_child(const shared_ptr<Cluster>& child) const
{
    return m_children.find(child) != m_children.end();
}

void Cluster::insert_parent(const shared_ptr<Cluster>& parent)
{
    m_parents.insert(parent);
}

void Cluster::remove_parent(const shared_ptr<Cluster>& parent)
{
    if (m_parents.find(parent) == m_parents.end())
    {
        throw ngraph_error("The parent cluster to remove is not a parent.");
    }
    m_parents.erase(parent);
}

void Cluster::remove_parent_if_exists(const shared_ptr<Cluster>& parent)
{
    m_parents.erase(parent);
}

bool Cluster::exist_parent(const shared_ptr<Cluster>& parent) const
{
    return m_parents.find(parent) != m_parents.end();
}

shared_ptr<Cluster> cluster_util::merge_clusters(const shared_ptr<Cluster>& src_cluster,
                                                 const shared_ptr<Cluster>& dst_cluster)
{
    // src_cluster and dst_cluster must have the same placement to merge
    if (src_cluster->get_placement() != dst_cluster->get_placement())
    {
        throw ngraph_error("Could not merge two clusters of different placment: " +
                           placement_to_string(src_cluster->get_placement()) + " and " +
                           placement_to_string(dst_cluster->get_placement()) + ".");
    }

    // The new cluster has all nodes from src_cluster and dst_cluster
    auto new_cluster = make_shared<Cluster>();
    for (auto node : src_cluster->get_nodes())
    {
        new_cluster->insert_node(node);
    }
    for (auto node : dst_cluster->get_nodes())
    {
        new_cluster->insert_node(node);
    }

    // Parents of src_cluster and dst_cluster should now have child new_cluster
    unordered_set<shared_ptr<Cluster>> all_parents;
    all_parents.insert(src_cluster->get_parents().begin(), src_cluster->get_parents().end());
    all_parents.insert(dst_cluster->get_parents().begin(), dst_cluster->get_parents().end());
    all_parents.erase(src_cluster);
    all_parents.erase(dst_cluster);
    for (shared_ptr<Cluster> parent : all_parents)
    {
        parent->remove_child_if_exists(src_cluster);
        parent->remove_child_if_exists(dst_cluster);
        parent->insert_child(new_cluster);
        new_cluster->insert_parent(parent);
    }

    // Children of src_cluster and dst_cluster should now have parent new_cluster
    unordered_set<shared_ptr<Cluster>> all_children;
    all_children.insert(src_cluster->get_children().begin(), src_cluster->get_children().end());
    all_children.insert(dst_cluster->get_children().begin(), dst_cluster->get_children().end());
    all_children.erase(src_cluster);
    all_children.erase(dst_cluster);
    for (shared_ptr<Cluster> child : all_children)
    {
        child->remove_parent_if_exists(src_cluster);
        child->remove_parent_if_exists(dst_cluster);
        child->insert_parent(new_cluster);
        new_cluster->insert_child(child);
    }

    src_cluster->clear_children();
    src_cluster->clear_parents();
    src_cluster->clear_nodes();
    dst_cluster->clear_children();
    dst_cluster->clear_parents();
    dst_cluster->clear_nodes();

    return new_cluster;
}

bool cluster_util::is_reachable_from_children_ptrs(const shared_ptr<Cluster>& src_cluster,
                                                   const shared_ptr<Cluster>& dst_cluster)
{
    unordered_set<shared_ptr<Cluster>> visited;
    deque<shared_ptr<Cluster>> stack;
    stack.push_front(src_cluster);

    while (!stack.empty())
    {
        shared_ptr<Cluster> curr_cluster = stack.front();
        stack.pop_front();
        if (visited.find(curr_cluster) != visited.end())
        {
            continue;
        }
        visited.insert(curr_cluster);
        if (curr_cluster == dst_cluster)
        {
            return true;
        }
        for (shared_ptr<Cluster> child_cluster : curr_cluster->get_children())
        {
            stack.push_front(child_cluster);
        }
    }

    return false;
}

bool cluster_util::is_edge_contractable(const shared_ptr<Cluster>& src_cluster,
                                        const shared_ptr<Cluster>& dst_cluster)
{
    bool rc;
    if (src_cluster->get_placement() != dst_cluster->get_placement())
    {
        rc = false;
    }
    else
    {
        // Contracting edge X->Y in a DAG forms a cycle iff after the removal of edge X->Y, Y is
        // still reachable from X.
        //
        // "=>": Proof that if "Contracting edge X->Y in a DAG forms a cycle", then "after the
        //       removal of edge X->Y, Y is still reachable from X".
        //
        // Let's call the original unaltered graph G. To contract edge X->Y, we
        // (1) Remove all Y's incoming and outgoing edges, including X->Y.
        // (2) Nodes set Is = { I | I != X, I != Y, I->Y in G, I->X not in G }.
        //     For all I in Is, add edge I->X.
        // (3) Nodes set Os = { O | O != X, O != Y, Y->O in G, X->O not in G }.
        //     For all O in Os, add edge X->O.
        // Is and Os must not have any nodes in common, because if they do have a common node N,
        // path N->Y->N would be in G which forms a cycle. The newly formed cycle C must contain
        // edge I->X or X->O, where I is in Is and O is in Os.
        //
        // Case 1: C contains both I->X and X->O, with I in Is and O in Os.
        //         Since path I->X->O is in C, path O->...->I is also in C. Path O->...->I does not
        //         contain any edges added by edge contraction, thus path O->...->I is also in G.
        //         Therefore path Y->O->...->I->Y is in G so G is not a DAG thus a contradiction.
        // Case 2: C contains I->X with I in Is, but does not contain X->O for all O in Os.
        //         Since I->X is in C, path X->...->I is also in C. Path X->...->I must not contain
        //         any edges added by edge contraction, and therefore, path X->...->I is in G and
        //         thus path X->...->I->Y is in G. Therefore, after removing edge X->Y from G, Y is
        //         still reachable from X by X->...->I->Y.
        // Case 3: C contains X->O with O in Os, but does not contain I->X for all I in Is.
        //         Since path X->O is in C, path O->...->X is also in C. Since C does not contain
        //         I->X, O->...->X does not contain any edges added by edge contraction, and so path
        //         O->...->X is in G. Therefore a cycle Y->O->...->X->Y is in G which contradicts
        //         with G is a DAG.
        //
        // "<=": Proof that if "after the removal of edge X->Y, Y is still reachable from X", then
        //       "contracting edge X->Y in a DAG forms a cycle".
        //
        // Assuming after the removal of edge X->Y, Y is still reachable from X from path
        // X->...->I->Y, then after contracting X->Y, a cycle X->...->I->X will be formed.
        //
        // Notes: Here edge means children pointer. Multiple edges from X->Y is considered as one
        //        edge since it doesn't matter in the context of node clustering.
        src_cluster->remove_child(dst_cluster);
        rc = !cluster_util::is_reachable_from_children_ptrs(src_cluster, dst_cluster);
        src_cluster->insert_child(dst_cluster);
    }
    return rc;
}

vector<shared_ptr<Cluster>> cluster_util::build_singleton_clusters(const shared_ptr<Function>& f)
{
    // Init: every cluster contains one node, and connect clusters with node's edges
    vector<shared_ptr<Cluster>> clusters;
    unordered_map<shared_ptr<Node>, shared_ptr<Cluster>> map_node_to_cluster;
    for (shared_ptr<Node> node : f->get_ordered_ops())
    {
        auto cluster = make_shared<Cluster>(unordered_set<shared_ptr<Node>>({node}));
        map_node_to_cluster[node] = cluster;
        clusters.push_back(cluster);
    }
    for (shared_ptr<Node> node : f->get_ordered_ops())
    {
        shared_ptr<Cluster> child_cluster = map_node_to_cluster.at(node);
        for (shared_ptr<Node> parent_node : node->get_input_ops())
        {
            shared_ptr<Cluster> parent_cluster = map_node_to_cluster.at(parent_node);
            parent_cluster->insert_child(child_cluster);
            child_cluster->insert_parent(parent_cluster);
        }
    }
    return clusters;
}

void cluster_util::merge_adjacent_clusters_pass(vector<shared_ptr<Cluster>>& clusters)
{
    // Contract cluster edges if the constraction does not form cycle(s)
    unordered_set<shared_ptr<Cluster>> unvisited_clusters(clusters.begin(), clusters.end());
    unordered_set<shared_ptr<Cluster>> candidate_clusters = unvisited_clusters;
    while (!unvisited_clusters.empty())
    {
        shared_ptr<Cluster> src_cluster = *(unvisited_clusters.begin());
        unvisited_clusters.erase(src_cluster);

        auto src_children = src_cluster->get_children();
        for (shared_ptr<Cluster> dst_cluster : src_children)
        {
            if (cluster_util::is_edge_contractable(src_cluster, dst_cluster))
            {
                shared_ptr<Cluster> new_custer =
                    cluster_util::merge_clusters(src_cluster, dst_cluster);
                unvisited_clusters.erase(dst_cluster);
                unvisited_clusters.insert(new_custer);
                candidate_clusters.erase(src_cluster);
                candidate_clusters.erase(dst_cluster);
                candidate_clusters.insert(new_custer);
                break;
            }
        }
    }
    clusters.clear();
    clusters.insert(clusters.end(), candidate_clusters.begin(), candidate_clusters.end());
}

void cluster_util::merge_disjoint_clusters_pass(vector<shared_ptr<Cluster>>& clusters)
{
    // Gather all possible placements
    set<Placement> all_placements;
    for (auto cluster : clusters)
    {
        all_placements.insert(cluster->get_placement());
    }

    // For each placement, merge clusters globally
    vector<shared_ptr<Cluster>> merged_clusters;
    for (auto target_placement : all_placements)
    {
        unordered_set<shared_ptr<Cluster>> unvisited_clusters;
        for (auto cluster : clusters)
        {
            if (cluster->get_placement() == target_placement)
            {
                unvisited_clusters.insert(cluster);
            }
        }
        unordered_set<shared_ptr<Cluster>> candidate_clusters = unvisited_clusters;
        while (!unvisited_clusters.empty())
        {
            shared_ptr<Cluster> src_cluster = *(unvisited_clusters.begin());
            unvisited_clusters.erase(src_cluster);

            for (shared_ptr<Cluster> dst_cluster : unvisited_clusters)
            {
                if (!cluster_util::is_reachable_from_children_ptrs(src_cluster, dst_cluster) &&
                    !cluster_util::is_reachable_from_children_ptrs(dst_cluster, src_cluster))
                {
                    shared_ptr<Cluster> new_custer =
                        cluster_util::merge_clusters(src_cluster, dst_cluster);
                    unvisited_clusters.erase(dst_cluster);
                    unvisited_clusters.insert(new_custer);
                    candidate_clusters.erase(src_cluster);
                    candidate_clusters.erase(dst_cluster);
                    candidate_clusters.insert(new_custer);
                    break;
                }
            }
        }
        merged_clusters.insert(
            merged_clusters.end(), candidate_clusters.begin(), candidate_clusters.end());
    }
    clusters = merged_clusters;
}

void cluster_util::topological_sort_clusters_pass(vector<shared_ptr<Cluster>>& clusters)
{
    // Kahn's algorithm, this also detects cycle iff it fail to complete
    deque<shared_ptr<Cluster>> independent_clusters;
    unordered_map<shared_ptr<Cluster>, size_t> map_cluster_to_num_dependencies;

    for (auto cluster : clusters)
    {
        size_t num_dependencies = cluster->get_parents().size();
        map_cluster_to_num_dependencies[cluster] = num_dependencies;
        if (num_dependencies == 0)
        {
            independent_clusters.push_back(cluster);
        }
    }

    vector<shared_ptr<Cluster>> sorted_clusters;
    while (!independent_clusters.empty())
    {
        auto independent_cluster = independent_clusters.front();
        independent_clusters.pop_front();
        sorted_clusters.push_back(independent_cluster);

        for (auto child_cluster : independent_cluster->get_children())
        {
            map_cluster_to_num_dependencies[child_cluster] -= 1;
            if (map_cluster_to_num_dependencies[child_cluster] == 0)
            {
                independent_clusters.push_back(child_cluster);
            }
        }
    }

    if (clusters.size() != sorted_clusters.size())
    {
        throw ngraph_error("Internal error: clusters.size() != sorted_clusters.size(): " +
                           to_string(clusters.size()) + " != " + to_string(sorted_clusters.size()) +
                           ". This is likely due to cycles in the Cluster graph.");
    }

    clusters = sorted_clusters;
}

void cluster_util::node_consistency_check_pass(const vector<shared_ptr<Cluster>>& clusters,
                                               const shared_ptr<Function>& f)
{
    // There shouldn't be any empty clusters
    for (auto cluster : clusters)
    {
        if (cluster->size() == 0)
        {
            throw ngraph_error("Cluster should not be empty.");
        }
    }

    // Create map[node]->cluster for checking
    // Also check node should and should only appear in one cluster
    unordered_map<shared_ptr<Node>, shared_ptr<Cluster>> map_node_to_cluster;
    for (auto cluster : clusters)
    {
        for (auto node : cluster->get_nodes())
        {
            if (map_node_to_cluster.find(node) != map_node_to_cluster.end())
            {
                throw ngraph_error("Internal error: node duplication in found in clusters: node " +
                                   node->get_name());
            }
            map_node_to_cluster[node] = cluster;
        }
    }

    // Clusters should contain same nodes as f
    unordered_set<shared_ptr<Node>> cluster_nodes_set;
    for (auto it : map_node_to_cluster)
    {
        cluster_nodes_set.insert(it.first);
    }
    unordered_set<shared_ptr<Node>> f_nodes_set;
    for (auto node : f->get_ordered_ops())
    {
        if (f_nodes_set.find(node) != f_nodes_set.end())
        {
            throw ngraph_error("Internal error: f contains duplicated node " + node->get_name());
        }
        f_nodes_set.insert(node);
    }
    if (f_nodes_set != cluster_nodes_set)
    {
        throw ngraph_error("Internal error: the nodes after clustering are not consistent.");
    }

    // Edges among clusters should be consistent with edges among nodes
    set<pair<shared_ptr<Cluster>, shared_ptr<Cluster>>> visited_cluster_edges_by_nodes;
    for (auto dst_node : f->get_ordered_ops())
    {
        for (auto src_node : dst_node->get_input_ops())
        {
            auto src_cluster = map_node_to_cluster.at(src_node);
            auto dst_cluster = map_node_to_cluster.at(dst_node);
            if (src_cluster != dst_cluster)
            {
                if (!src_cluster->exist_child(dst_cluster))
                {
                    throw ngraph_error("Node edge " + src_node->get_name() + "->" +
                                       dst_node->get_name() + " exists, but edge " +
                                       src_cluster->get_debug_node_names() + "->" +
                                       dst_cluster->get_debug_node_names() + " does not.");
                }
                visited_cluster_edges_by_nodes.insert(make_pair(src_cluster, dst_cluster));
            }
        }
    }
    for (auto src_cluster : clusters)
    {
        for (auto dst_cluster : src_cluster->get_children())
        {
            auto cluster_edge = make_pair(src_cluster, dst_cluster);
            if (visited_cluster_edges_by_nodes.find(cluster_edge) ==
                visited_cluster_edges_by_nodes.end())
            {
                throw ngraph_error("Excess cluster edge detected.");
            }
        }
    }

    // Consistency of child and parent pointers of clusters
    // Child -> parent
    for (auto src_cluster : clusters)
    {
        for (auto dst_cluster : src_cluster->get_children())
        {
            if (!dst_cluster->exist_parent(src_cluster))
            {
                throw ngraph_error("dst is a child of src, but src is not a parent of dst");
            }
        }
    }
    // Parent -> child
    for (auto dst_cluster : clusters)
    {
        for (auto src_cluster : dst_cluster->get_parents())
        {
            if (!src_cluster->exist_child(dst_cluster))
            {
                throw ngraph_error("src is a parent of dst, but dst is not a child of src");
            }
        }
    }
}

vector<shared_ptr<Cluster>> cluster_util::split_function_to_clusters(const shared_ptr<Function>& f)
{
    // Init node cluster, every cluster contains one node
    vector<shared_ptr<Cluster>> clusters = cluster_util::build_singleton_clusters(f);
    node_consistency_check_pass(clusters, f);

    // Run cluster optimization passes
    // TODO: Currently runs node_consistency_check_pass for safety
    //       We only need to run once at the end
    cluster_util::merge_adjacent_clusters_pass(clusters);
    cluster_util::node_consistency_check_pass(clusters, f);

    cluster_util::merge_disjoint_clusters_pass(clusters);
    cluster_util::node_consistency_check_pass(clusters, f);

    cluster_util::topological_sort_clusters_pass(clusters);
    cluster_util::node_consistency_check_pass(clusters, f);

    return clusters;
}

//*****************************************************************************
// Copyright 2019 Intel Corporation
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
/// \file
/// Implementation of CPURuntimeContextCG related utilities.
///

#include "pregenerated_src.hpp"

#include "ngraph/code_writer.hpp"

void ngraph::runtime::cpu::emit_runtime_context(CodeWriter& writer)
{
    writer << R"(
struct CPURuntimeContextCG
{
    std::unique_ptr<tbb::flow::graph> m_tbb_graph;
    std::unique_ptr<tbb::global_control> m_tbb_gcontrol;

    CPURuntimeContextCG() { init_tbb(); }
    ~CPURuntimeContextCG() { cleanup_tbb(); }

private:
    inline void init_tbb()
    {
        if (std::getenv("NGRAPH_CPU_USE_TBB"))
        {
            m_tbb_graph.reset(new tbb::flow::graph);
            const char* env_parallelism = std::getenv("NGRAPH_INTER_OP_PARALLELISM");
            const int parallelism = env_parallelism == nullptr ? 1 : std::atoi(env_parallelism);
            m_tbb_gcontrol.reset(
                new tbb::global_control(tbb::global_control::max_allowed_parallelism, parallelism));
        }
    }

    inline void cleanup_tbb()
    {
        if (std::getenv("NGRAPH_CPU_USE_TBB"))
        {
            // Delete nodes in m_tbb_graph.
            m_tbb_graph->wait_for_all();
            std::vector<tbb::flow::graph_node*> to_be_deleted;
            for (auto it = m_tbb_graph->begin(); it != m_tbb_graph->end(); it++)
            {
                to_be_deleted.push_back(&*it);
            }
            for (auto* node : to_be_deleted)
            {
                delete node;
            }
        }
    }
};

extern "C" CPURuntimeContextCG* init_cg_ctx()
{
    return new CPURuntimeContextCG;
}

extern "C" void destroy_cg_ctx(CPURuntimeContextCG* cg_ctx)
{
    delete cg_ctx;
}
)";
}

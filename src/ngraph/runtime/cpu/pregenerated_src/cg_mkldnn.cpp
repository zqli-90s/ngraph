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
/// Implementation of MKLDNN code generation utilities.
///

#include "pregenerated_src.hpp"

#include "ngraph/code_writer.hpp"
#include "ngraph/runtime/cpu/mkldnn_emitter.hpp"

using namespace ngraph::codegen;

void ngraph::runtime::cpu::emit_mkldnn_utils(CodeWriter& writer,
                                             const MKLDNNEmitter& mkldnn_emitter)
{
    // Not generated if there are no mkldnn nodes.
    if (!mkldnn_emitter.get_mkldnn_primitives().size())
        return;

    writer << R"(
struct MKLDNNUtils
{
    static void set_memory_ptr(CPURuntimeContextCG* cg_ctx, size_t primitive_index, void* ptr)
    {
        auto* primitive = static_cast<mkldnn::memory*>(cg_ctx->m_mkldnn_primitives[primitive_index]);
        primitive->set_data_handle(ptr);
    }

    static void invoke_primitive(CPURuntimeContextCG* cg_ctx, size_t primitive_index)
    {
        mkldnn::stream strm(mkldnn::stream::kind::eager);
        // TODO: Can we just propagate the exception here instead of throwing ngraph_error?
        //try
        //{
            strm.submit({*cg_ctx->m_mkldnn_primitives[primitive_index]}).wait();
        //}
        //catch (const mkldnn::error& err)
        //{
        //    throw ngraph_error("Could not run mkdnn primitive " + err.message);
        //}
    }
};
)";
}

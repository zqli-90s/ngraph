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
/// This file contains utilities to dump pre-generated source code that is needed in codegen mode.
///

#pragma once

#include "ngraph/code_writer.hpp"

#include <vector>

// Forward decls
namespace ngraph
{
    class Node;
}

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            /// Generate CPURuntimeContextCG. This class is used to hold runtime information of
            /// the execution of kernels in codegen mode.
            void emit_runtime_context(CodeWriter& writer,
                                      const std::vector<const Node*>& mkldnn_nodes);

            // MKLDNN Utils

            /// Generate MKLDNN utilities used to set up MKLDNN execution environment.
            void emit_mkldnn_utils(CodeWriter& writer);
        }
    }
}

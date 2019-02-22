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
#include "ngraph/node.hpp"
//#include "ngraph/op/abs.hpp"
//#include "ngraph/op/acos.hpp"
//#include "ngraph/op/add.hpp"
//#include "ngraph/op/all.hpp"
//#include "ngraph/op/allreduce.hpp"
//#include "ngraph/op/and.hpp"
//#include "ngraph/op/any.hpp"
//#include "ngraph/op/argmax.hpp"
//#include "ngraph/op/argmin.hpp"
//#include "ngraph/op/asin.hpp"
//#include "ngraph/op/atan.hpp"
//#include "ngraph/op/avg_pool.hpp"
//#include "ngraph/op/batch_norm.hpp"
//#include "ngraph/op/broadcast.hpp"
//#include "ngraph/op/ceiling.hpp"
//#include "ngraph/op/concat.hpp"
//#include "ngraph/op/constant.hpp"
//#include "ngraph/op/convert.hpp"
#include "ngraph/op/convolution.hpp"
//#include "ngraph/op/cos.hpp"
//#include "ngraph/op/cosh.hpp"
//#include "ngraph/op/dequantize.hpp"
//#include "ngraph/op/divide.hpp"
//#include "ngraph/op/dot.hpp"
//#include "ngraph/op/embedding_lookup.hpp"
//#include "ngraph/op/equal.hpp"
//#include "ngraph/op/exp.hpp"
//#include "ngraph/op/experimental/generate_mask.hpp"
//#include "ngraph/op/experimental/quantized_avg_pool.hpp"
//#include "ngraph/op/experimental/quantized_conv.hpp"
//#include "ngraph/op/experimental/quantized_conv_bias.hpp"
//#include "ngraph/op/experimental/quantized_conv_relu.hpp"
//#include "ngraph/op/experimental/quantized_max_pool.hpp"
//#include "ngraph/op/floor.hpp"
//#include "ngraph/op/get_output_element.hpp"
//#include "ngraph/op/greater.hpp"
//#include "ngraph/op/greater_eq.hpp"
//#include "ngraph/op/less.hpp"
//#include "ngraph/op/less_eq.hpp"
//#include "ngraph/op/log.hpp"
//#include "ngraph/op/lrn.hpp"
//#include "ngraph/op/max.hpp"
//#include "ngraph/op/max_pool.hpp"
//#include "ngraph/op/maximum.hpp"
//#include "ngraph/op/min.hpp"
//#include "ngraph/op/minimum.hpp"
//#include "ngraph/op/multiply.hpp"
//#include "ngraph/op/negative.hpp"
//#include "ngraph/op/not.hpp"
//#include "ngraph/op/not_equal.hpp"
//#include "ngraph/op/one_hot.hpp"
//#include "ngraph/op/op.hpp"
//#include "ngraph/op/or.hpp"
//#include "ngraph/op/pad.hpp"
//#include "ngraph/op/parameter.hpp"
//#include "ngraph/op/power.hpp"
//#include "ngraph/op/product.hpp"
//#include "ngraph/op/quantize.hpp"
//#include "ngraph/op/relu.hpp"
//#include "ngraph/op/replace_slice.hpp"
//#include "ngraph/op/reshape.hpp"
//#include "ngraph/op/result.hpp"
//#include "ngraph/op/reverse.hpp"
//#include "ngraph/op/reverse_sequence.hpp"
//#include "ngraph/op/select.hpp"
//#include "ngraph/op/sign.hpp"
//#include "ngraph/op/sin.hpp"
//#include "ngraph/op/sinh.hpp"
//#include "ngraph/op/slice.hpp"
//#include "ngraph/op/softmax.hpp"
//#include "ngraph/op/sqrt.hpp"
//#include "ngraph/op/subtract.hpp"
//#include "ngraph/op/sum.hpp"
//#include "ngraph/op/tan.hpp"
//#include "ngraph/op/tanh.hpp"
//#include "ngraph/op/topk.hpp"

using namespace ngraph;

using OpStringMap = std::unordered_map<std::type_index, const std::string>;

#define TI(x) std::type_index(typeid(x))

static const OpStringMap mkldnn_init_dispatcher{
    //    {TI(ngraph::op::Add), "UNDEFINED()"},
    //#ifdef NGRAPH_DISTRIBUTED_ENABLE
    //    {TI(ngraph::op::AllReduce), "UNDEFINED()"},
    //#endif
    //    {TI(ngraph::op::MatmulBias), "UNDEFINED()"},
    //    {TI(ngraph::op::Dot), "UNDEFINED()"},
    //    {TI(ngraph::op::Multiply), "UNDEFINED()"},
    //    {TI(ngraph::op::Parameter), "UNDEFINED()"},
    //    {TI(ngraph::op::Abs), "UNDEFINED()"},
    //    {TI(ngraph::op::Any), "UNDEFINED()"},
    //    {TI(ngraph::op::All), "UNDEFINED()"},
    //    {TI(ngraph::op::BatchDot), "UNDEFINED()"},
    //    {TI(ngraph::op::Concat), "UNDEFINED()"},
    //    {TI(ngraph::op::Divide), "UNDEFINED()"},
    //    {TI(ngraph::op::Equal), "UNDEFINED()"},
    //    {TI(ngraph::op::GetOutputElement), "UNDEFINED()"},
    //    {TI(ngraph::op::Greater), "UNDEFINED()"},
    //    {TI(ngraph::op::GreaterEq), "UNDEFINED()"},
    //    {TI(ngraph::op::Less), "UNDEFINED()"},
    //    {TI(ngraph::op::LessEq), "UNDEFINED()"},
    //    {TI(ngraph::op::Log), "UNDEFINED()"},
    //    {TI(ngraph::op::Maximum), "UNDEFINED()"},
    //    {TI(ngraph::op::Minimum), "UNDEFINED()"},
    //    {TI(ngraph::op::Negative), "UNDEFINED()"},
    //    {TI(ngraph::op::NotEqual), "UNDEFINED()"},
    //    {TI(ngraph::op::Power), "UNDEFINED()"},
    //    {TI(ngraph::op::Select), "UNDEFINED()"},
    //    {TI(ngraph::op::Subtract), "UNDEFINED()"},
    //    {TI(ngraph::op::Broadcast), "UNDEFINED()"},
    //    {TI(ngraph::op::Convert), "UNDEFINED()"},
    //    {TI(ngraph::op::Constant), "UNDEFINED()"},
    //    {TI(ngraph::op::Reshape), "UNDEFINED()"},
    //    {TI(ngraph::op::Sign), "UNDEFINED()"},
    //    {TI(ngraph::op::Slice), "UNDEFINED()"},
    //    {TI(ngraph::op::Sum), "UNDEFINED()"},
    //    {TI(ngraph::op::EmbeddingLookup), "UNDEFINED()"},
    //    {TI(ngraph::op::Exp), "UNDEFINED()"},
    //    {TI(ngraph::op::Sin), "UNDEFINED()"},
    //    {TI(ngraph::op::Sinh), "UNDEFINED()"},
    //    {TI(ngraph::op::Cos), "UNDEFINED()"},
    //    {TI(ngraph::op::Cosh), "UNDEFINED()"},
    //    {TI(ngraph::op::Tan), "UNDEFINED()"},
    //    {TI(ngraph::op::Tanh), "UNDEFINED()"},
    //    {TI(ngraph::op::TopK), "UNDEFINED()"},
    //    {TI(ngraph::op::Asin), "UNDEFINED()"},
    //    {TI(ngraph::op::ArgMin), "UNDEFINED()"},
    //    {TI(ngraph::op::ArgMax), "UNDEFINED()"},
    //    {TI(ngraph::op::Acos), "UNDEFINED()"},
    //    {TI(ngraph::op::Atan), "UNDEFINED()"},
    //    {TI(ngraph::op::ReplaceSlice), "UNDEFINED()"},
    //    {TI(ngraph::op::UpdateSlice), "UNDEFINED()"},
    //    {TI(ngraph::op::OneHot), "UNDEFINED()"},
    //    {TI(ngraph::op::Floor), "UNDEFINED()"},
    //    {TI(ngraph::op::Ceiling), "UNDEFINED()"},
    //    {TI(ngraph::op::Sqrt), "UNDEFINED()"},
    {TI(ngraph::op::Convolution), "UNDEFINED()"},
    //    {TI(ngraph::op::ConvolutionBackpropFilters), "UNDEFINED()"},
    //    {TI(ngraph::op::ConvolutionBackpropData), "UNDEFINED()"},
    //    {TI(ngraph::op::GroupConvolution), "UNDEFINED()"},
    //    {TI(ngraph::op::ConvolutionBias), "UNDEFINED()"},
    //    {TI(ngraph::op::QuantizedConvolutionBias), "UNDEFINED()"},
    //    {TI(ngraph::op::QuantizedConvolutionBiasAdd), "UNDEFINED()"},
    //    {TI(ngraph::op::QuantizedConvolutionBiasSignedAdd), "UNDEFINED()"},
    //    {TI(ngraph::op::ConvolutionRelu), "UNDEFINED()"},
    //    {TI(ngraph::op::QuantizedConvolution), "UNDEFINED()"},
    //    {TI(ngraph::op::QuantizedConvolutionRelu), "UNDEFINED()"},
    //    {TI(ngraph::op::ConvolutionBiasAdd), "UNDEFINED()"},
    //    // conv+bias backprop for data share the same implementation as ConvolutionBackpropData
    //    {TI(ngraph::op::ConvolutionBiasBackpropFiltersBias), "UNDEFINED()"},
    //    {TI(ngraph::runtime::cpu::op::ConvertLayout), "UNDEFINED()"},
    //    {TI(ngraph::op::Not), "UNDEFINED()"},
    //    {TI(ngraph::op::MaxPool), "UNDEFINED()"},
    //    {TI(ngraph::op::QuantizedMaxPool), "UNDEFINED()"},
    //    {TI(ngraph::op::QuantizedAvgPool), "UNDEFINED()"},
    //    {TI(ngraph::op::MaxPoolWithIndices), "UNDEFINED()"},
    //    {TI(ngraph::op::Reverse), "UNDEFINED()"},
    //    {TI(ngraph::op::ReverseSequence), "UNDEFINED()"},
    //    {TI(ngraph::op::Result), "UNDEFINED()"},
    //    {TI(ngraph::op::AvgPool), "UNDEFINED()"},
    //    {TI(ngraph::op::AvgPoolBackprop), "UNDEFINED()"},
    //    {TI(ngraph::op::Pad), "UNDEFINED()"},
    //    {TI(ngraph::op::BatchNormTraining), "UNDEFINED()"},
    //    {TI(ngraph::op::BatchNormInference), "UNDEFINED()"},
    //    {TI(ngraph::op::BatchNormTrainingRelu), "UNDEFINED()"},
    //    {TI(ngraph::op::BatchNormInferenceRelu), "UNDEFINED()"},
    //    {TI(ngraph::op::BatchNormTrainingBackprop), "UNDEFINED()"},
    //    {TI(ngraph::op::BoundedRelu), "UNDEFINED()"},
    //    {TI(ngraph::op::Lstm), "UNDEFINED()"},
    //    {TI(ngraph::op::MaxPoolBackprop), "UNDEFINED()"},
    //    {TI(ngraph::op::MaxPoolWithIndicesBackprop), "UNDEFINED()"},
    //    {TI(ngraph::op::Product), "UNDEFINED()"},
    //    {TI(ngraph::op::Max), "UNDEFINED()"},
    //    {TI(ngraph::op::Min), "UNDEFINED()"},
    //    {TI(ngraph::op::Relu), "UNDEFINED()"},
    //    {TI(ngraph::op::ReluBackprop), "UNDEFINED()"},
    //    {TI(ngraph::op::Rnn), "UNDEFINED()"},
    //    {TI(ngraph::op::Sigmoid), "UNDEFINED()"},
    //    {TI(ngraph::op::SigmoidMultiply), "UNDEFINED()"},
    //    {TI(ngraph::op::SigmoidMultiplyBackprop), "UNDEFINED()"},
    //    {TI(ngraph::op::Softmax), "UNDEFINED()"},
    //    {TI(ngraph::op::SigmoidBackprop), "UNDEFINED()"},
    //    {TI(ngraph::op::And), "UNDEFINED()"},
    //    {TI(ngraph::op::Or), "UNDEFINED()"},
    //    {TI(ngraph::op::LeakyRelu), "UNDEFINED()"},
    //    {TI(ngraph::runtime::cpu::op::LoopKernel), "UNDEFINED()"},
    //    {TI(ngraph::op::LRN), "UNDEFINED()"},
    //    {TI(ngraph::op::GenerateMask), "UNDEFINED()"},
    //    {TI(ngraph::op::ConvolutionAdd), "UNDEFINED()"},
    //    {TI(ngraph::op::Quantize), "UNDEFINED()"},
    //    {TI(ngraph::op::Dequantize), "UNDEFINED()"},
    //    {TI(ngraph::op::GroupConvolutionBias), "UNDEFINED()"},
};

// Emit MKLDNN members of CPURuntimeContextCG. It includes:
//   1) MKLDNNPrimitives: enum that assigns an ID to each MKL primitive used in the code.
//   2) m_mkl_primitive_idxs: holds the descriptor for each member of MKLDNNPrimitives.
static CodeWriter& emit_mkldnn_members(CodeWriter& writer,
                                       const std::vector<const Node*>& mkldnn_nodes)
{
    // Enum is not generated if there are no mkldnn nodes.
    if (!mkldnn_nodes.size())
        return writer;

    writer << "\n";
    writer.indent++;
    writer << "// MLKDNN members\n";
    writer << "enum MKLDNNPrimitives { ";

    bool first_node = true;
    for (const Node* node : mkldnn_nodes)
    {
        std::string op_name = node->description();
        std::transform(op_name.begin(), op_name.end(), op_name.begin(), ::toupper);
        writer << "NG_MKLDNN_" << op_name;

        if (first_node)
        {
            // Initialize the first enum member to zero.
            writer << " = 0";
            first_node = false;
        }
        writer << ", ";
    }

    writer << "NG_MKLDNN_NUM_PRIMITIVES };\n";
    writer << "std::array<size_t, NG_MKLDNN_NUM_PRIMITIVES> m_mkl_primitive_idxs;\n";
    writer << "MKLDNNEmitter m_mkldnn_emitter;";
    writer.indent--;

    return writer;
}

// Emit 'init_mkldnn' utility, which initializes MKLDNN environment.
static CodeWriter& emit_init_mkldnn(CodeWriter& writer,
                                    const std::vector<const Node*>& mkldnn_nodes)
{
    // Not generated if there no mkldnn nodes.
    if (!mkldnn_nodes.size())
        return writer;

    writer << "\n";
    writer.indent++;
    writer << "inline void init_mkldnn()\n";
    writer.block_begin();

    for (size_t i = 0, end = mkldnn_nodes.size(); i < end; ++i)
    {
        auto mkl_init_func_it = mkldnn_init_dispatcher.find(TI(*mkldnn_nodes[i]));
        NGRAPH_ASSERT(mkl_init_func_it != mkldnn_init_dispatcher.end())
            << "Unexpected node for MKLDNN.";

        writer << "m_mkl_primitive_idxs[" << i << "] = m_mkldnn_emitter."
               << mkl_init_func_it->second << ";\n";
    }

    writer.block_end();
    writer.indent--;
}

void ngraph::runtime::cpu::emit_runtime_context(CodeWriter& writer,
                                                const std::vector<const Node*>& mkldnn_nodes)
{
    bool has_mkldnn_nodes = mkldnn_nodes.size();

    writer << R"(
struct CPURuntimeContextCG
{
    std::unique_ptr<tbb::flow::graph> m_tbb_graph;
    std::unique_ptr<tbb::global_control> m_tbb_gcontrol;
)";
    emit_mkldnn_members(writer, mkldnn_nodes);
    writer << R"(
    CPURuntimeContextCG() { init_tbb(); }
    ~CPURuntimeContextCG() { cleanup_tbb(); }
)";

    if (has_mkldnn_nodes)
    {
        writer << R"(
    inline mkldnn::primitive* get_mkldnn_primitive(size_t primitive_index)
    {
        return &*mkldnn_primitives[primitive_index];
    }
)";
    }

    writer << R"(
private:
    std::vector<std::unique_ptr<mkldnn::primitive>> mkldnn_primitives;     

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
)";
    emit_init_mkldnn(writer, mkldnn_nodes);
    writer << R"(};

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

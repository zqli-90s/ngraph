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

#include <memory>

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/softmax.hpp"
#include "util/random.hpp"
#include "util/autodiff/backprop_function.hpp"

using namespace std;
using namespace ngraph;

TEST(shape, test_shape_size)
{
    ASSERT_EQ(1, shape_size(Shape{}));
    ASSERT_EQ(2 * 3 * 5, shape_size(Shape{2, 3, 5}));
}

TEST(shape, test_shape_strides)
{
    ASSERT_EQ(Strides{}, row_major_strides(Shape{}));
    ASSERT_EQ(Strides{1}, row_major_strides(Shape{3}));
    ASSERT_EQ((Strides{7, 1}), row_major_strides(Shape{2, 7}));
    ASSERT_EQ((Strides{84, 12, 1}), row_major_strides(Shape{5, 7, 12}));
}

TEST(shape, test_partial_shape_mnist_mlp)
{
//    PartialShape data_batch_shape{PartialShape::dynamic()};
//    PartialShape data_batch_shape{Dimension::dynamic(), 1, 28, 28};
//    PartialShape reshape_1_shape{Dimension::dynamic(), 1*28*28};
    Shape data_batch_shape{2, 1, 28, 28};
    Shape reshape_1_shape{2, 1*28*28};
    Shape weight_1_shape{1*28*28, 128};
    Shape weight_2_shape{128, 64};
    Shape weight_3_shape{64, 10};
    
    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto weight_1 = make_shared<op::Parameter>(element::f32, weight_1_shape);
    auto weight_2 = make_shared<op::Parameter>(element::f32, weight_2_shape);
    auto weight_3 = make_shared<op::Parameter>(element::f32, weight_3_shape);

    auto reshape_1 = make_shared<op::Reshape>(data_batch, AxisVector{0,1,2,3}, reshape_1_shape);
    auto fc_1 = make_shared<op::Dot>(reshape_1, weight_1);
    auto act_1 = make_shared<op::Relu>(fc_1);
    auto fc_2 = make_shared<op::Dot>(act_1, weight_2);
    auto act_2 = make_shared<op::Relu>(fc_2);
    auto fc_3 = make_shared<op::Dot>(act_2, weight_3);
    auto softmax_1 = make_shared<op::Softmax>(fc_3, AxisSet{1});
    
    auto f = make_shared<Function>(NodeVector{softmax_1}, ParameterVector{data_batch, weight_1, weight_2, weight_3});
    
    test::Uniform<float> rng(-0.5f, 0.5f);
    vector<vector<float>> fprop_args;
    for (shared_ptr<op::Parameter> param : f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        fprop_args.push_back(tensor_val);
    }
    auto fprop_results = execute(f, fprop_args, "INTERPRETER");
    std::cout << "fprop results:" << std::endl;
    for (auto& result : fprop_results) {
        std::cout << vector_to_string(result) << std::endl;    
    } 
    
    auto df = autodiff::backprop_function(f);
    vector<vector<float>> bprop_args;
    for (shared_ptr<op::Parameter> param : df->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        bprop_args.push_back(tensor_val);
    }
    auto bprop_results = execute(df, bprop_args, "INTERPRETER");
    std::cout << "bprop results:" << std::endl;
    for (auto& result : bprop_results) {
        std::cout << vector_to_string(result) << std::endl;    
    }
}

#if 0
TEST(shape, test_partial_shape_mnist_cnn)
{
//    PartialShape data_batch_shape{PartialShape::dynamic()};
    Shape data_batch_shape{2,1,28,28};
    Shape weight_1_shape{128, 1*28*28};
    Shape weight_2_shape{64, 128};
    Shape weight_3_shape{10, 64};
    
    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto weight_1 = make_shared<op::Parameter>(element::f32, weight_1_shape);
    auto weight_2 = make_shared<op::Parameter>(element::f32, weight_2_shape);
    auto weight_3 = make_shared<op::Parameter>(element::f32, weight_3_shape);

    auto fc_1 = make_shared<op::Dot>(data_batch, weight_1);
    auto act_1 = make_shared<op::Relu>(fc_1);
    auto fc_2 = make_shared<op::Dot>(act_1, weight_2);
    auto act_2 = make_shared<op::Relu>(fc_2);
    auto fc_3 = make_shared<op::Dot>(act_2, weight_3);
    auto softmax = make_shared<op::Softmax>(fc_3);
    
    auto f = make_shared<Function>(NodeVector{softmax}, ParameterVector{data_batch, weight_1, weight_2, weight_3});
    
    test::Uniform<float> rng(0.0f, 1.0f);
    vector<vector<float>> fprop_args;
    for (shared_ptr<op::Parameter> param : f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        fprop_args.push_back(tensor_val);
    }
    auto int_results = execute(f, fprop_args, "INTERPRETER");
    
    auto df = autodiff::backprop_function(f);
    vector<vector<float>> bprop_args;
    for (shared_ptr<op::Parameter> param : df->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        bprop_args.push_back(tensor_val);
    }
    auto bprop_results = execute(df, bprop_args, "INTERPRETER");
}
    
#endif

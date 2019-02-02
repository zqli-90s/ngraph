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

#include <algorithm>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "ngraph/function.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

TEST(tensor, size)
{
    pass::Manager pass_manager;

    pass_manager.register_pass<pass::Liveness>();

    {
        auto arg0 = make_shared<op::Parameter>(element::f32, Shape{2, 3});
        auto add = make_shared<op::Add>(arg0, arg0);
        auto f0 = make_shared<Function>(add, ParameterVector{arg0});

        pass_manager.run_passes(f0);

        auto& outputs = arg0->get_outputs();
        ASSERT_EQ(1, outputs.size());
        descriptor::Tensor& output = outputs[0].get_tensor();
        EXPECT_EQ(2 * 3 * 4, output.size());
    }

    {
        auto arg0 = make_shared<op::Parameter>(element::f32, Shape{});
        auto add = make_shared<op::Add>(arg0, arg0);
        auto f0 = make_shared<Function>(add, ParameterVector{arg0});

        pass_manager.run_passes(f0);

        auto& outputs = arg0->get_outputs();
        ASSERT_EQ(1, outputs.size());
        descriptor::Tensor& output = outputs[0].get_tensor();
        EXPECT_EQ(1 * 4, output.size());
    }

    {
        auto arg0 = make_shared<op::Parameter>(element::f32, Shape{1});
        auto add = make_shared<op::Add>(arg0, arg0);
        auto f0 = make_shared<Function>(add, ParameterVector{arg0});

        pass_manager.run_passes(f0);

        auto& outputs = arg0->get_outputs();
        ASSERT_EQ(1, outputs.size());
        descriptor::Tensor& output = outputs[0].get_tensor();
        EXPECT_EQ(1 * 4, output.size());
    }
}

template <typename T>
void test_read_write(const vector<T>& x)
{
    auto backend = runtime::Backend::create("INTERPRETER");

    auto a = backend->create_tensor(element::from<T>(), Shape{2, x.size()});

    vector<T> result(2 * x.size());

    a->write(&x[0], 0, x.size() * sizeof(T));
    copy(x.begin(), x.end(), result.begin());
    a->write(&x[0], x.size() * sizeof(T), x.size() * sizeof(T));
    copy(x.begin(), x.end(), result.begin() + x.size());

    vector<T> af_vector(2 * x.size());
    a->read(af_vector.data(), 0, af_vector.size() * sizeof(T));
    ASSERT_EQ(af_vector, result);

    vector<T> result1(x.size());
    vector<T> result2(x.size());
    copy(result.begin() + 1, result.begin() + 1 + x.size(), result1.begin());
    a->read(&result2[0], sizeof(T), sizeof(T) * x.size());
    ASSERT_EQ(result1, result2);
}

#if defined(NGRAPH_INTERPRETER_ENABLE)
TEST(tensor, read_write)
{
    test_read_write<float>({1.0, 3.0, 5.0});
    test_read_write<int64_t>({-1, 2, 4});
}
#endif

TEST(tensor, output_flag)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::Liveness>();

    auto arg0 = make_shared<op::Parameter>(element::f32, Shape{1});
    auto add = make_shared<op::Add>(arg0, arg0);
    auto f0 = make_shared<Function>(add, ParameterVector{arg0});

    pass_manager.run_passes(f0);

    for (size_t i = 0; i < f0->get_output_size(); ++i)
    {
        EXPECT_TRUE(f0->get_output_op(i)->is_output());
    }
}

template <typename T>
string pretty(T v)
{
    stringstream ss;
    ss.imbue(locale(""));
    ss << v;
    return ss.str();
}

string get_type(element::Type t)
{
    switch (t.get_type_enum())
    {
    case element::Type_t::undefined: return "element::undefined";
    case element::Type_t::dynamic: return "element::dynamic";
    case element::Type_t::boolean: return "element::boolean";
    case element::Type_t::bf16: return "element::bf16";
    case element::Type_t::f32: return "element::f32";
    case element::Type_t::f64: return "element::f64";
    case element::Type_t::i8: return "element::i8";
    case element::Type_t::i16: return "element::i16";
    case element::Type_t::i32: return "element::i32";
    case element::Type_t::i64: return "element::i64";
    case element::Type_t::u8: return "element::u8";
    case element::Type_t::u16: return "element::u16";
    case element::Type_t::u32: return "element::u32";
    case element::Type_t::u64: return "element::u64";
    }
}

TEST(tensor, transfer_rate)
{
    stopwatch timer;
    Shape shape{256, 3, 224, 224};
    // Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    // auto B = make_shared<op::Not>(A);
    auto f = make_shared<Function>(A, ParameterVector{A});

    vector<pair<Shape, element::Type>> parameters = {
        {Shape{}, element::i64},
        {Shape{32, 224, 224, 3}, element::f32},
        {Shape{7, 7, 3, 64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{1, 1, 64, 64}, element::f32},
        {Shape{1, 1, 64, 256}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{3, 3, 64, 64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{1, 1, 64, 256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{1, 1, 256, 64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{3, 3, 64, 64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{1, 1, 64, 256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{1, 1, 256, 64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{3, 3, 64, 64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{1, 1, 64, 256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{1, 1, 256, 128}, element::f32},
        {Shape{1, 1, 256, 512}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{3, 3, 128, 128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{1, 1, 128, 512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{1, 1, 512, 128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{3, 3, 128, 128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{1, 1, 128, 512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{1, 1, 512, 128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{3, 3, 128, 128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{1, 1, 128, 512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{1, 1, 512, 128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{3, 3, 128, 128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{1, 1, 128, 512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{1, 1, 512, 256}, element::f32},
        {Shape{1, 1, 512, 1024}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{3, 3, 256, 256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{1, 1, 256, 1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1, 1, 1024, 256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{3, 3, 256, 256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{1, 1, 256, 1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1, 1, 1024, 256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{3, 3, 256, 256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{1, 1, 256, 1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1, 1, 1024, 256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{3, 3, 256, 256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{1, 1, 256, 1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1, 1, 1024, 256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{3, 3, 256, 256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{1, 1, 256, 1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1, 1, 1024, 256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{3, 3, 256, 256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{1, 1, 256, 1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1, 1, 1024, 512}, element::f32},
        {Shape{1, 1, 1024, 2048}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{3, 3, 512, 512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{1, 1, 512, 2048}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{1, 1, 2048, 512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{3, 3, 512, 512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{1, 1, 512, 2048}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{1, 1, 2048, 512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{3, 3, 512, 512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{1, 1, 512, 2048}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{2048, 1001}, element::f32},
        {Shape{1001}, element::f32},
        {Shape{32}, element::i32},
    };
    vector<pair<Shape, element::Type>> results = {
        {Shape{}, element::boolean},
        {Shape{5}, element::boolean},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{32, 64, 56, 56}, element::f32},
        {Shape{32, 64, 56, 56}, element::f32},
        {Shape{}, element::boolean},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{}, element::boolean},
        {Shape{}, element::boolean},
        {Shape{}, element::boolean},
        {Shape{}, element::boolean},
        {Shape{32, 64, 56, 56}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{32, 256, 56, 56}, element::f32},
        {Shape{32, 64, 56, 56}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{32, 64, 56, 56}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{32, 256, 56, 56}, element::f32},
        {Shape{32, 64, 56, 56}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{32, 64, 56, 56}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{32, 256, 56, 56}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{32, 128, 58, 58}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{32, 128, 28, 28}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{4}, element::i32},
        {Shape{4}, element::i32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{32, 512, 28, 28}, element::f32},
        {Shape{32, 128, 28, 28}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{32, 128, 28, 28}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{32, 512, 28, 28}, element::f32},
        {Shape{32, 128, 28, 28}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{32, 128, 28, 28}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{32, 512, 28, 28}, element::f32},
        {Shape{32, 128, 28, 28}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{32, 128, 28, 28}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{32, 512, 28, 28}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{32, 256, 30, 30}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{32, 256, 14, 14}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{4}, element::i32},
        {Shape{4}, element::i32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{32, 1024, 14, 14}, element::f32},
        {Shape{32, 256, 14, 14}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{32, 256, 14, 14}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{32, 1024, 14, 14}, element::f32},
        {Shape{32, 256, 14, 14}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{32, 256, 14, 14}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{32, 1024, 14, 14}, element::f32},
        {Shape{32, 256, 14, 14}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{32, 256, 14, 14}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{32, 1024, 14, 14}, element::f32},
        {Shape{32, 256, 14, 14}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{32, 256, 14, 14}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{32, 1024, 14, 14}, element::f32},
        {Shape{32, 256, 14, 14}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{32, 256, 14, 14}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{32, 1024, 14, 14}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{32, 512, 16, 16}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{32, 512, 7, 7}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{4}, element::i32},
        {Shape{4}, element::i32},
        {Shape{2048}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{32, 2048, 7, 7}, element::f32},
        {Shape{32, 512, 7, 7}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{32, 512, 7, 7}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{32, 2048, 7, 7}, element::f32},
        {Shape{32, 512, 7, 7}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{32, 512, 7, 7}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{2}, element::i32},
        {Shape{4}, element::i32},
        {Shape{1}, element::i32},
        {Shape{1}, element::i32},
        {Shape{}, element::f32},
        {Shape{32, 1001}, element::f32},
        {Shape{32, 2048}, element::f32},
        {Shape{}, element::f32},
        {Shape{32, 2048, 7, 7}, element::f32},
        {Shape{32, 2048, 7, 7}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{32, 512, 7, 7}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{32, 512, 7, 7}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{32, 2048, 7, 7}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{32, 512, 7, 7}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{32, 512, 7, 7}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{32, 2048, 7, 7}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{32, 2048, 7, 7}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{2048}, element::f32},
        {Shape{4}, element::i32},
        {Shape{32, 512, 7, 7}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{4}, element::i32},
        {Shape{32, 512, 14, 14}, element::f32},
        {Shape{32, 512, 14, 14}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{32, 1024, 14, 14}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{32, 256, 14, 14}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{32, 256, 14, 14}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{32, 1024, 14, 14}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{32, 256, 14, 14}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{32, 256, 14, 14}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{32, 1024, 14, 14}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{32, 256, 14, 14}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{32, 256, 14, 14}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{32, 1024, 14, 14}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{32, 256, 14, 14}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{32, 256, 14, 14}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{32, 1024, 14, 14}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{32, 256, 14, 14}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{32, 256, 14, 14}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{32, 1024, 14, 14}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{32, 1024, 14, 14}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{1024}, element::f32},
        {Shape{4}, element::i32},
        {Shape{32, 256, 14, 14}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{4}, element::i32},
        {Shape{32, 256, 28, 28}, element::f32},
        {Shape{32, 256, 28, 28}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{32, 512, 28, 28}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{32, 128, 28, 28}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{32, 128, 28, 28}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{32, 512, 28, 28}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{32, 128, 28, 28}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{32, 128, 28, 28}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{32, 512, 28, 28}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{32, 128, 28, 28}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{32, 128, 28, 28}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{32, 512, 28, 28}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{32, 512, 28, 28}, element::f32},
        {Shape{512}, element::f32},
        {Shape{512}, element::f32},
        {Shape{4}, element::i32},
        {Shape{32, 128, 28, 28}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{4}, element::i32},
        {Shape{32, 128, 56, 56}, element::f32},
        {Shape{32, 128, 56, 56}, element::f32},
        {Shape{128}, element::f32},
        {Shape{128}, element::f32},
        {Shape{32, 256, 56, 56}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{32, 64, 56, 56}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{32, 64, 56, 56}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{32, 256, 56, 56}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{32, 64, 56, 56}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{32, 64, 56, 56}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{32, 256, 56, 56}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{32, 256, 56, 56}, element::f32},
        {Shape{256}, element::f32},
        {Shape{256}, element::f32},
        {Shape{32, 64, 56, 56}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{32, 64, 56, 56}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{32, 64, 112, 112}, element::f32},
        {Shape{32, 64, 112, 112}, element::f32},
        {Shape{64}, element::f32},
        {Shape{64}, element::f32},
        {Shape{32, 3, 230, 230}, element::f32},
        {Shape{4}, element::i32},
    };

    /// This is the code used to generate the parameter/result arrays listed here
    // vector<string> files = {"/nfs/site/home/rhkimbal/r50/tf_function_ngraph_cluster_481.json",
    //                         "/nfs/site/home/rhkimbal/r50/tf_function_ngraph_cluster_482.json",
    //                         "/nfs/site/home/rhkimbal/r50/tf_function_ngraph_cluster_487.json",
    //                         "/nfs/site/home/rhkimbal/r50/tf_function_ngraph_cluster_496.json",
    //                         "/nfs/site/home/rhkimbal/r50/tf_function_ngraph_cluster_502.json"};
    // for (string file : files)
    // {
    //     NGRAPH_INFO << file;
    //     auto graph = deserialize(file);
    //     size_t total_in = 0;
    //     size_t total_out = 0;
    //     cout << "vector<pair<Shape, element::Type>> parameters = {\n";
    //     for (auto node : graph->get_parameters())
    //     {
    //         size_t element_size = node->get_element_type().size();
    //         size_t element_count = shape_size(node->get_shape());
    //         size_t tensor_size = element_count * element_size;
    //         total_in += tensor_size;
    //         cout << "{" << node->get_shape() << ", " << get_type(node->get_element_type())
    //              << "},\n";
    //     }
    //     cout << "};\n";
    //     cout << "vector<pair<Shape, element::Type>> results = {\n";
    //     for (auto node : graph->get_results())
    //     {
    //         size_t element_size = node->get_element_type().size();
    //         size_t element_count = shape_size(node->get_shape());
    //         size_t tensor_size = element_count * element_size;
    //         total_out += tensor_size;
    //         cout << "{" << node->get_shape() << ", " << get_type(node->get_element_type())
    //              << "},\n";
    //     }
    //     cout << "};\n";
    //     NGRAPH_INFO << graph->get_parameters().size() << ", " << pretty(total_in);
    //     NGRAPH_INFO << graph->get_results().size() << ", " << pretty(total_out);
    // }

    {
        auto backend = runtime::Backend::create("GPU");
        timer.start();
        for (const pair<Shape, element::Type>& p : parameters)
        {
            const Shape& shape = p.first;
            const element::Type& type = p.second;
            shared_ptr<runtime::Tensor> a = backend->create_tensor(type, shape);
            vector<float> a_data(shape_size(shape));
            a->write(a_data.data(), 0, a_data.size() * type.size());
        }
        timer.stop();
        cout << "copy params " << timer.get_microseconds() << "us\n";
        for (const pair<Shape, element::Type>& p : results)
        {
        }
    }

    ofstream out("simple_transfer.json");
    string s = serialize(f, 4);
    out << s;

    auto backend = runtime::Backend::create("GPU");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    vector<float> a_data(shape_size(shape));
    timer.start();
    a->write(a_data.data(), 0, a_data.size() * sizeof(float));
    timer.stop();
    cout.imbue(locale(""));

    cout << "copy to device " << timer.get_microseconds() << "us\n";

    vector<float> r_data(shape_size(shape));
    timer.start();
    memcpy(r_data.data(), a_data.data(), shape_size(shape));
    timer.stop();
    cout << "memcpy " << timer.get_microseconds() << "us\n";

    // shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    // auto handle = backend->compile(f);
    // backend->call(handle, {result}, {a});
    // result->read(r_data.data(), 0, r_data.size() * sizeof(float));
}

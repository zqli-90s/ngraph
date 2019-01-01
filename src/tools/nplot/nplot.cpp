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

#include <fstream>
#include <iomanip>

#include "ngraph/serializer.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"

using namespace std;
using namespace ngraph;

int main(int argc, char** argv)
{
    string input;
    string output;
    bool failed = false;

    for (size_t i = 1; i < argc; i++)
    {
        string arg = argv[i];
         if (arg == "-i" || arg == "--input")
        {
            input = argv[++i];
        }
        else if (arg == "-o" || arg == "--output")
        {
            output = argv[++i];
        }
    }

    if (!input.empty() && !output.empty())
    {
        cout << "input:  "  << input << endl;
        cout << "output: "  << output << endl;
        ifstream in(input);
        if (in)
        {
            auto function = deserialize(in);
            pass::Manager pm;
            pm.register_pass<ngraph::pass::VisualizeTree>(output);
            pm.run_passes(function);
        }
    }
    else 
    {
        cout << R"###(
DESCRIPTION
    Benchmark ngraph json model with given backend.

SYNOPSIS
        nbench [-f <filename>] [-b <backend>] [-i <iterations>]

OPTIONS
        -f|--file                 Serialized model file
        -b|--backend              Backend to use (default: CPU)
        -d|--directory            Directory to scan for models. All models are benchmarked.
        -i|--iterations           Iterations (default: 10)
        -s|--statistics           Display op stastics
        -v|--visualize            Visualize a model (WARNING: requires GraphViz installed)
        --timing_detail           Gather detailed timing
        -w|--warmup_iterations    Number of warm-up iterations
        --no_copy_data            Disable copy of input/result data every iteration
)###";
        return 1;
    }

    return 0;
}

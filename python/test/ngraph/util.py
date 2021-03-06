# ******************************************************************************
# Copyright 2018-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

import numpy as np
import ngraph as ng

from string import ascii_uppercase

import test


def _get_numpy_dtype(scalar):
    return np.array([scalar]).dtype


def get_runtime():
    """Return runtime object."""
    return ng.runtime(backend_name=test.BACKEND_NAME)


def run_op_node(input_data, op_fun, *args):
    """Run computation on node performing `op_fun`.

    `op_fun` has to accept a node as an argument.

    :param input_data: The input data for performed computation.
    :param op_fun: The function handler for operation we want to carry out.
    :param args: The arguments passed to operation we want to carry out.
    :return: The result from computations.
    """
    runtime = get_runtime()
    comp_args = []
    op_fun_args = []
    comp_inputs = []
    for idx, data in enumerate(input_data):
        if np.isscalar(data):
            op_fun_args.append(ng.constant(data, _get_numpy_dtype(data)))
        else:
            node = ng.parameter(data.shape, name=ascii_uppercase[idx], dtype=data.dtype)
            op_fun_args.append(node)
            comp_args.append(node)
            comp_inputs.append(data)
    op_fun_args.extend(args)
    node = op_fun(*op_fun_args)
    computation = runtime.computation(node, *comp_args)
    return computation(*comp_inputs)


def run_op_numeric_data(input_data, op_fun, *args):
    """Run computation on node performing `op_fun`.

    `op_fun` has to accept a scalar or an array.

    :param input_data: The input data for performed computation.
    :param op_fun: The function handler for operation we want to carry out.
    :param args: The arguments passed to operation we want to carry out.
    :return: The result from computations.
    """
    runtime = get_runtime()
    node = op_fun(input_data, *args)
    computation = runtime.computation(node)
    return computation()

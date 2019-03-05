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

#pragma once

#include "ngraph/node_vector.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Takes a slice of an input tensor, i.e., the sub-tensor that resides within a bounding box, optionally with stride.
        class DynSlice : public Op
        {
        public:
            /// \brief Constructs a tensor slice operation.
            ///
            /// \param arg The tensor to be sliced.
            /// \param begin The axiswise begin of the slice (inclusive).
            /// \param end The axiswise end of the slice (exclusive).
            /// \param strides The slicing strides; for example, strides of `{n,m}` means to take
            ///                every nth row and every mth column of the input matrix.
            Slice(const std::shared_ptr<Node>& arg,
                  const std::shared_ptr<Node>& begin,
                  const std::shared_ptr<Node>& end,
                  const std::shared_ptr<Node>& strides);

            /// \brief Constructs a tensor slice operation with unit strides; i.e., every element inside the bounding box will be copied to the output slice.
            ///
            /// \param arg The tensor to be sliced.
            /// \param begin The axiswise begin of the slice (inclusive).
            /// \param end The axiswise end of the slice (exclusive).
            Slice(const std::shared_ptr<Node>& arg,
                  const std::shared_ptr<Node>& begin,
                  const std::shared_ptr<Node>& end)

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            /// \return The inclusive begin coordinates.
            const std::shared_ptr<Node>& get_begin() const { return m_begin; }
            /// \return The exclusive end coordinates.
            const std::shared_ptr<Node>& get_end() const { return m_end; }
            /// \return The slicing strides.
            const std::shared_ptr<Node>& get_strides() const { return m_strides; }
        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;
            void validate_and_infer_types() override;

            std::shared_ptr<Node> m_begin;
            std::shared_ptr<Node> m_end;
            std::shared_ptr<Node> m_strides;
        };
    }
}

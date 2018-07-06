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

#pragma once

#include <map>
#include <memory>
#include <string>

// Class for things which have attributes.
//
// TODO(amprocte): fold this into Node, or move it above Node, or something like that.

#include "ngraph/attribute.hpp"
#include "ngraph/attribute_map.hpp"

namespace ngraph
{
    class HasAttributes
    {
    public:
        // TODO(amprocte): make the parameter into a unique_ptr?
        HasAttributes(AttributeMap* attribute_map)
            : m_attribute_map(std::unique_ptr<AttributeMap>(attribute_map))
        {
        }

        template<typename T>
        T& get_attribute(std::string attribute_name)
        {
            // TODO(amprocte): get rid of dynamic_cast here?
            // TODO(amprocte): we're relying on "at" and "dynamic_cast" to throw exceptions here,
            // and those exceptions may not be very user-friendly.
            Attribute* p_attr = m_attribute_map->at(attribute_name);
            return *(dynamic_cast<T*>(p_attr));
        }

        template<typename T>
        T& get_boxed_attribute(std::string attribute_name)
        {
            // TODO(amprocte): get rid of dynamic_cast here?
            // TODO(amprocte): we're relying on "at" and "dynamic_cast" to throw exceptions here,
            // and those exceptions may not be very user-friendly.
            Attribute* p_attr = m_attribute_map->at(attribute_name);
            BoxedAttribute<T>* p_boxed = dynamic_cast<BoxedAttribute<T>*>(p_attr);
            return p_boxed->get_ref();
        }

    private:
        std::unique_ptr<AttributeMap> m_attribute_map;
    };
}

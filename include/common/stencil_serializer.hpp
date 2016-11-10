/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/

#pragma once

#ifdef CXX11_ENABLED

#include <algorithm>
#include <string>
#include <vector>
#include "defs.hpp"

namespace gridtools {

    template < class SerializerType >
    class stencil_serializer {
        std::string m_stencil_name;       /**< Name of the stencil currently being serialized */
        SerializerType *m_serializer;     /**< Non-owning pointer to the Serializer */
        int_t m_stage_id;                 /**< Keep track of the number of stages */
        int_t m_stencil_invocation_count; /**< Keep track of stencil invocation count i.e calls to run(...) */

      public:
        stencil_serializer(const std::string &stencil_name, SerializerType &serializer)
            : m_stencil_name(stencil_name), m_serializer(&serializer), m_stage_id(0) {

            // Check how many times the stencil was invoked
            std::string savepoint_name = m_stencil_name + "__out";

            m_stencil_invocation_count = 0;
            for (const auto &sp : serializer.savepoints()) {
                if (sp.name() == savepoint_name)
                    m_stencil_invocation_count = sp.meta_info().template as< int_t >("invocation_count") + 1;
            }

            // Register the stencil name within the serializer (if not present yet)
            std::vector< std::string > stencil_name_vec;
            if (serializer.global_meta_info().has_key("stencils")) {
                stencil_name_vec = serializer.global_meta_info().template as< std::vector< std::string > >("stencils");
                serializer.global_meta_info().erase("stencils");
            }

            stencil_name_vec.push_back(m_stencil_name);
            auto it = std::unique(stencil_name_vec.begin(), stencil_name_vec.end());
            stencil_name_vec.resize(std::distance(stencil_name_vec.begin(), it));

            serializer.global_meta_info().insert("stencils", stencil_name_vec);
        }

        /**
         * \brief Access Serializer
         */
        SerializerType &get_serializer() { return *m_serializer; }

        /**
         * \brief Access name of the stencil currently being serialized
         */
        const std::string &get_stencil_name() const { return m_stencil_name; }

        /**
         * \brief Get the current stage id and increment it afterwards
         */
        int_t get_and_increment_stage_id() { return m_stage_id++; }

        /**
         * \brief Get current invocation count (i.e calls to run(...))
         */
        const int_t &stencil_invocation_count() const { return m_stencil_invocation_count; }
    };

} // namespace gridtools

#endif

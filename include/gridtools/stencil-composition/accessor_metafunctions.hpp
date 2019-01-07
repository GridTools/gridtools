/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
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

#include <type_traits>

#include "../common/defs.hpp"
#include "../meta/at.hpp"
#include "../meta/length.hpp"
#include "../meta/type_traits.hpp"
#include "./accessor_fwd.hpp"

namespace gridtools {
    template <class Accessor, class = void>
    struct is_accessor_readonly : std::false_type {};

    template <class Accessor>
    struct is_accessor_readonly<Accessor, enable_if_t<Accessor::intent == enumtype::in>> : std::true_type {};

    /* Is written is actually "can be written", since it checks if not read only.*/
    template <class Accessor>
    struct is_accessor_written : negation<is_accessor_readonly<Accessor>> {};

    template <class Accessor>
    struct accessor_index {
        GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), GT_INTERNAL_ERROR);
        using type = typename Accessor::index_t;
    };

    namespace _impl {
        template <size_t ID, class ArgsMap>
        constexpr ushort_t get_remap_accessor_id() {
            GRIDTOOLS_STATIC_ASSERT(meta::length<ArgsMap>::value != 0, GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT(ID < meta::length<ArgsMap>::value, GT_INTERNAL_ERROR);
            return meta::lazy::at_c<ArgsMap, ID>::type::value;
        }
    } // namespace _impl
} // namespace gridtools

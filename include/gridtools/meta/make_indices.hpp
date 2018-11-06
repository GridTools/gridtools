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

#include <cstddef>

#include "defs.hpp"
#include "iseq_to_list.hpp"
#include "length.hpp"
#include "macros.hpp"
#include "utility.hpp"

namespace gridtools {
    namespace meta {
        GT_META_LAZY_NAMESPASE {
            /**
             *  Make a list of integral constants of indices from 0 to N
             */
            template <std::size_t N>
            GT_META_DEFINE_ALIAS(make_indices_c, iseq_to_list, make_index_sequence<N>);

            template <class N>
            GT_META_DEFINE_ALIAS(make_indices, iseq_to_list, make_index_sequence<N::value>);

            /**
             *  Make a list of integral constants of indices from 0 to length< List >
             */
            template <class List>
            GT_META_DEFINE_ALIAS(make_indices_for, iseq_to_list, make_index_sequence<length<List>::value>);
        }
#if !GT_BROKEN_TEMPLATE_ALIASES
        template <std::size_t N>
        using make_indices_c = typename lazy::iseq_to_list<make_index_sequence<N>>::type;
        template <class N>
        using make_indices = typename lazy::iseq_to_list<make_index_sequence<N::value>>::type;
        template <class List>
        using make_indices_for = typename lazy::iseq_to_list<make_index_sequence<length<List>::value>>::type;
#endif
    } // namespace meta
} // namespace gridtools

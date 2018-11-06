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
#include <type_traits>

#include "defs.hpp"
#include "first.hpp"
#include "macros.hpp"
#include "make_indices.hpp"
#include "mp_find.hpp"
#include "second.hpp"
#include "zip.hpp"

namespace gridtools {
    namespace meta {
        /**
         *   Take Nth element of the List
         */
        GT_META_LAZY_NAMESPASE {
            template <class List, std::size_t N>
            struct at_c;

            template <class List>
            struct at_c<List, 0> : first<List> {};

            template <class List>
            struct at_c<List, 1> : second<List> {};

            template <class List, std::size_t N>
            struct at_c : second<typename mp_find<typename zip<typename make_indices_for<List>::type, List>::type,
                              std::integral_constant<std::size_t, N>>::type> {};

            template <class List, class N>
            GT_META_DEFINE_ALIAS(at, at_c, (List, N::value));
        }
#if !GT_BROKEN_TEMPLATE_ALIASES
        // 'direct' versions of lazy functions
        template <class List, class N>
        using at = typename lazy::at_c<List, N::value>::type;
        template <class List, std::size_t N>
        using at_c = typename lazy::at_c<List, N>::type;
#endif
    } // namespace meta
} // namespace gridtools

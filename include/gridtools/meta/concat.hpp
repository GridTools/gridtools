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

#include "combine.hpp"
#include "list.hpp"
#include "macros.hpp"

namespace gridtools {
    namespace meta {
        /**
         *  Concatenate lists
         */
        GT_META_LAZY_NAMESPASE {
            template <class...>
            struct concat;
        }
        GT_META_DELEGATE_TO_LAZY(concat, class... Lists, Lists...);
        GT_META_LAZY_NAMESPASE {
            template <>
            struct concat<> : list<> {};
            template <template <class...> class L, class... Ts>
            struct concat<L<Ts...>> {
                using type = L<Ts...>;
            };
            template <template <class...> class L1, class... T1s, template <class...> class L2, class... T2s>
            struct concat<L1<T1s...>, L2<T2s...>> {
                using type = L1<T1s..., T2s...>;
            };

            template <class L1, class L2, class L3, class... Lists>
            struct concat<L1, L2, L3, Lists...> : combine<meta::concat, list<L1, L2, L3, Lists...>> {};
        }
    } // namespace meta
} // namespace gridtools

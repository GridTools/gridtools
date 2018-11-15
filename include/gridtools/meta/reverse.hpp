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

#include "macros.hpp"
#include "push_back.hpp"

namespace gridtools {
    namespace meta {
        /**
         *   reverse algorithm.
         *   Complexity is O(N)
         *   Making specializations for the first M allows to divide complexity by M.
         *   At a moment M = 4 (in boost::mp11 implementation it is 10).
         *   For the optimizers: fill free to add more specializations if needed.
         */
        GT_META_LAZY_NAMESPACE {
            template <class>
            struct reverse;

            template <template <class...> class L>
            struct reverse<L<>> {
                using type = L<>;
            };
            template <template <class...> class L, class T>
            struct reverse<L<T>> {
                using type = L<T>;
            };
            template <template <class...> class L, class T0, class T1>
            struct reverse<L<T0, T1>> {
                using type = L<T1, T0>;
            };
            template <template <class...> class L, class T0, class T1, class T2>
            struct reverse<L<T0, T1, T2>> {
                using type = L<T2, T1, T0>;
            };
            template <template <class...> class L, class T0, class T1, class T2, class T3>
            struct reverse<L<T0, T1, T2, T3>> {
                using type = L<T3, T2, T1, T0>;
            };
            template <template <class...> class L, class T0, class T1, class T2, class T3, class T4>
            struct reverse<L<T0, T1, T2, T3, T4>> {
                using type = L<T4, T3, T2, T1, T0>;
            };
            template <template <class...> class L, class T0, class T1, class T2, class T3, class T4, class... Ts>
            struct reverse<L<T0, T1, T2, T3, T4, Ts...>>
                : push_back<typename reverse<L<Ts...>>::type, T4, T3, T2, T1, T0> {};
        }
        GT_META_DELEGATE_TO_LAZY(reverse, class List, List);
    } // namespace meta
} // namespace gridtools

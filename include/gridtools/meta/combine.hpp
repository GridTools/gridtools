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

#include "curry_fun.hpp"
#include "defs.hpp"
#include "drop_front.hpp"
#include "length.hpp"
#include "macros.hpp"

namespace gridtools {
    namespace meta {
        /**
         *   Applies binary function to the elements of the list.
         *
         *   For example:
         *     combine<f>::apply<list<t1, t2, t3, t4, t5, t6, t7>> === f<f<f<t1, t2>, f<t3, f4>>, f<f<t5, t6>, t7>>
         *
         *   Complexity is amortized O(N), the depth of template instantiation is O(log(N))
         */
        GT_META_LAZY_NAMESPASE {
            template <template <class...> class, class...>
            struct combine;
        }
        GT_META_DELEGATE_TO_LAZY(combine, (template <class...> class F, class... Args), (F, Args...));

        GT_META_LAZY_NAMESPASE {
            template <template <class...> class F, class List, std::size_t N>
            struct combine_impl {
                static_assert(N > 0, "N in combine_impl<F, List, N> must be positive");
                static constexpr std::size_t m = N / 2;
                using type = GT_META_CALL(F,
                    (typename combine_impl<F, List, m>::type,
                        typename combine_impl<F, typename drop_front_c<m, List>::type, N - m>::type));
            };
            template <template <class...> class F, template <class...> class L, class T, class... Ts>
            struct combine_impl<F, L<T, Ts...>, 1> {
                using type = T;
            };
            template <template <class...> class F, template <class...> class L, class T1, class T2, class... Ts>
            struct combine_impl<F, L<T1, T2, Ts...>, 2> {
                using type = GT_META_CALL(F, (T1, T2));
            };
            template <template <class...> class F,
                template <class...> class L,
                class T1,
                class T2,
                class T3,
                class... Ts>
            struct combine_impl<F, L<T1, T2, T3, Ts...>, 3> {
                using type = GT_META_CALL(F, (T1, GT_META_CALL(F, (T2, T3))));
            };
            template <template <class...> class F,
                template <class...> class L,
                class T1,
                class T2,
                class T3,
                class T4,
                class... Ts>
            struct combine_impl<F, L<T1, T2, T3, T4, Ts...>, 4> {
                using type = GT_META_CALL(F, (GT_META_CALL(F, (T1, T2)), GT_META_CALL(F, (T3, T4))));
            };
            template <template <class...> class F>
            struct combine<F> {
                using type = curry_fun<meta::combine, F>;
            };
            template <template <class...> class F, class List>
            struct combine<F, List> : combine_impl<F, List, length<List>::value> {};
        }
    } // namespace meta
} // namespace gridtools

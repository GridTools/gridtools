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

#include "curry_fun.hpp"
#include "fold.hpp"
#include "list.hpp"
#include "macros.hpp"
#include "push_back.hpp"
#include "rename.hpp"

namespace gridtools {
    namespace meta {
        /**
         *   Transform `Lists` by applying `F` element wise.
         *
         *   I.e the first element of resulting list would be `F<first_from_l0, first_froml1, ...>`;
         *   the second would be `F<second_from_l0, ...>` and so on.
         *
         *   For N lists M elements each complexity is O(N). I.e for one list it is O(1).
         */
        GT_META_LAZY_NAMESPACE {
            template <template <class...> class, class...>
            struct transform;
        }
        GT_META_DELEGATE_TO_LAZY(transform, (template <class...> class F, class... Args), (F, Args...));

        GT_META_LAZY_NAMESPACE {
            template <template <class...> class F>
            struct transform<F> {
                using type = curry_fun<meta::transform, F>;
            };
            template <template <class...> class F, template <class...> class L, class... Ts>
            struct transform<F, L<Ts...>> {
                using type = L<GT_META_CALL(F, Ts)...>;
            };
            template <template <class...> class F,
                template <class...> class L1,
                class... T1s,
                template <class...> class L2,
                class... T2s>
            struct transform<F, L1<T1s...>, L2<T2s...>> {
                using type = L1<GT_META_CALL(F, (T1s, T2s))...>;
            };

            /**
             *   Takes `2D array` of types (i.e. list of lists where inner lists are the same length) and do
             *   trasposition. Example:
             *   a<b<void, void*, void**>, b<int, int*, int**>> => b<a<void, int>, a<void*, int*>, a<void**, int**>>
             */
            template <class>
            struct transpose;
            template <template <class...> class L>
            struct transpose<L<>> {
                using type = list<>;
            };
            template <template <class...> class Outer, template <class...> class Inner, class... Ts, class... Inners>
            struct transpose<Outer<Inner<Ts...>, Inners...>>
                : lfold<transform<meta::push_back>::type::apply, Inner<Outer<Ts>...>, list<Inners...>> {};

            // transform, generic version
            template <template <class...> class F, class List, class... Lists>
            struct transform<F, List, Lists...>
                : transform<rename<F>::type::template apply, typename transpose<list<List, Lists...>>::type> {};
        }
        GT_META_DELEGATE_TO_LAZY(transpose, class List, List);
    } // namespace meta
} // namespace gridtools

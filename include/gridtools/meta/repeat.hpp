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
#include "list.hpp"
#include "macros.hpp"

namespace gridtools {
    namespace meta {
        /**
         *  Produce a list of N identical elements
         */
        GT_META_LAZY_NAMESPASE {
            template <class List, bool Rem>
            struct repeat_impl_expand;

            template <class... Ts>
            struct repeat_impl_expand<list<Ts...>, false> {
                using type = list<Ts..., Ts...>;
            };

            template <class T, class... Ts>
            struct repeat_impl_expand<list<T, Ts...>, true> {
                using type = list<T, T, T, Ts..., Ts...>;
            };

            template <size_t N, class T>
            struct repeat_c {
                using type = typename repeat_impl_expand<typename repeat_c<N / 2, T>::type, N % 2>::type;
            };

            template <class T>
            struct repeat_c<0, T> {
                using type = list<>;
            };

            template <class T>
            struct repeat_c<1, T> {
                using type = list<T>;
            };

            template <class N, class T>
            GT_META_DEFINE_ALIAS(repeat, repeat_c, (N::value, T));
        }
#if !GT_BROKEN_TEMPLATE_ALIASES
        template <size_t N, class T>
        using repeat_c = typename lazy::repeat_c<N, T>::type;
        template <class N, class T>
        using repeat = typename lazy::repeat_c<N::value, T>::type;
#endif
    } // namespace meta
} // namespace gridtools

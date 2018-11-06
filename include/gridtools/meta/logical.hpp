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

#include "list.hpp"
#include "macros.hpp"
#include "rename.hpp"
#include "repeat.hpp"
#include "transform.hpp"
#include "type_traits.hpp"

namespace gridtools {
    namespace meta {

        /**
         *  NVCC bug workaround: sizeof... works incorrectly within template alias context.
         */
#ifdef __NVCC__
        template <class... Ts>
        struct sizeof_3_dots : std::integral_constant<size_t, sizeof...(Ts)> {};

#define GT_SIZEOF_3_DOTS(Ts) ::gridtools::meta::sizeof_3_dots<Ts...>::value
#else
#define GT_SIZEOF_3_DOTS(Ts) sizeof...(Ts)
#endif

        /**
         *  C++17 drop-offs
         *
         *  Note on conjunction_fast and disjunction_fast are like std counter parts but:
         *    - short-circuiting is not implemented as required by C++17 standard
         *    - amortized complexity is O(1) because of it [in terms of the number of template instantiations].
         */
        template <class... Ts>
        GT_META_DEFINE_ALIAS(conjunction_fast,
            std::is_same,
            (list<std::integral_constant<bool, Ts::value>...>,
                GT_META_CALL(repeat_c, (GT_SIZEOF_3_DOTS(Ts), std::true_type))));

        template <class... Ts>
        GT_META_DEFINE_ALIAS(disjunction_fast, negation, conjunction_fast<negation<Ts>...>);

/**
 *   all/some elements in lists are true
 */
#if GT_BROKEN_TEMPLATE_ALIASES
        template <class List>
        struct all : rename<conjunction_fast, List>::type {};

        template <class List>
        struct any : rename<disjunction_fast, List>::type {};
#else
        template <class List>
        using all = rename<conjunction_fast, List>;

        template <class List>
        using any = rename<disjunction_fast, List>;
#endif

        /**
         *  All elements satisfy predicate
         */
        template <template <class...> class Pred, class List>
        GT_META_DEFINE_ALIAS(all_of, all, (GT_META_CALL(transform, (Pred, List))));

        /**
         *  Some element satisfy predicate
         */
        template <template <class...> class Pred, class List>
        GT_META_DEFINE_ALIAS(any_of, any, (GT_META_CALL(transform, (Pred, List))));
    } // namespace meta
} // namespace gridtools

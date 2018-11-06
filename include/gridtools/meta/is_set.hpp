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

#include "dedup.hpp"
#include "id.hpp"
#include "internal/inherit.hpp"
#include "macros.hpp"
#include "type_traits.hpp"

namespace gridtools {
    namespace meta {
        /**
         *   True if the template parameter is type list which elements are all different
         */
        template <class>
        struct is_set : std::false_type {};

        template <template <class...> class L, class... Ts>
        struct is_set<L<Ts...>> : std::is_same<L<Ts...>, GT_META_CALL(dedup, L<Ts...>)> {};

        /**
         *   is_set_fast evaluates to std::true_type if the parameter is a set.
         *   If parameter is not a type list, predicate evaluates to std::false_type.
         *   Compilation fails if the parameter is a type list with duplicated elements.
         *
         *   Its OK to use this predicate in static asserts and not OK in sfinae enablers.
         */
        template <class, class = void>
        struct is_set_fast : std::false_type {};

        template <template <class...> class L, class... Ts>
        struct is_set_fast<L<Ts...>, void_t<decltype(internal::inherit<lazy::id<Ts>...>{})>> : std::true_type {};
    } // namespace meta
} // namespace gridtools

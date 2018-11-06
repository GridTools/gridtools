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

#include "at.hpp"
#include "id.hpp"
#include "list.hpp"
#include "macros.hpp"

namespace gridtools {
    namespace meta {
        template <size_t>
        struct placeholder;

        using _1 = placeholder<0>;
        using _2 = placeholder<1>;
        using _3 = placeholder<2>;
        using _4 = placeholder<3>;
        using _5 = placeholder<4>;
        using _6 = placeholder<5>;
        using _7 = placeholder<6>;
        using _8 = placeholder<7>;
        using _9 = placeholder<8>;
        using _10 = placeholder<9>;

        template <class Arg, class... Params>
        struct replace_placeholders_impl : lazy::id<Arg> {};

        template <size_t I, class... Params>
        struct replace_placeholders_impl<placeholder<I>, Params...> : lazy::at_c<list<Params...>, I> {};

        /**
         *  bind for functions
         */
        template <template <class...> class F, class... BoundArgs>
        struct bind {
            template <class... Params>
            GT_META_DEFINE_ALIAS(apply, F, (typename replace_placeholders_impl<BoundArgs, Params...>::type...));
        };
    } // namespace meta
} // namespace gridtools

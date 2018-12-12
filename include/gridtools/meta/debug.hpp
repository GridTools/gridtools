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

#include <boost/preprocessor.hpp>

/**
 *  This macro expands to the code snippet that generates a compiler error that refers to the type(s) `x`
 *
 *  Works also with parameter packs. I.e you can both `GT_META_PRINT_TYPE(SomeType)` and
 * `GT_META_PRINT_TYPE(SomeTypes...)`
 */
#define GT_META_PRINT_TYPE(x) static_assert(::gridtools::meta::debug::type<BOOST_PP_REMOVE_PARENS(x)>::_, "")

/**
 *  This macro expands to the code snippet that generates a compiler error that refers to the compile time value(s) of
 * the integral type  `x`
 *
 *  Works also with parameter packs. I.e you can both `GT_META_PRINT_VALUE(SomeValue)` and
 * `GT_META_PRINT_VALUE(SomeValues...)`
 */
#define GT_META_PRINT_VALUE(x)                                                                                \
    static_assert(                                                                                            \
        ::gridtools::meta::debug::value<decltype(::gridtools::meta::debug::first(BOOST_PP_REMOVE_PARENS(x))), \
            BOOST_PP_REMOVE_PARENS(x)>::_,                                                                    \
        "")

namespace gridtools {
    namespace meta {
        namespace debug {
            template <class T>
            T first(T, ...);

            template <class...>
            struct type {};
            template <class T, T...>
            struct value {};
        } // namespace debug
    }     // namespace meta
} // namespace gridtools

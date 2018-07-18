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

#include "defs.hpp"
#include "generic_metafunctions/gt_integer_sequence.hpp"
#include "generic_metafunctions/meta.hpp"
#include "tuple_util.hpp"

namespace gridtools {
    namespace impl_ {
        template <typename Res>
        struct permute_to_impl;

        template <template <typename...> class Res, typename... Elems>
        struct permute_to_impl<Res<Elems...>> {
            template <typename Src>
            Res<Elems...> operator()(Src &&src) {
                using src_t = typename std::decay<Src>::type;
                return Res<Elems...>{
                    tuple_util::get<meta::st_position<src_t, Elems>::value>(std::forward<Src>(src))...};
            }
        };
    } // namespace impl_

    /** \ingroup common
     * \defgroup permute_to Permute
     *
     *  For each type in Res find the element in src of the same type, place those elements in correct order and
     *  construct the Res instance from them.
     *
     *  This utility is handy when we have all elements of the Res, but not in the right order.
     *
     *  Requirements:
     *      - Res and Src should model tuple-like sequence;
     *      - Res type should have a ctor from a tuple-like sequence;
     *      - all types from the Res should present in the Src;
     *
     *  Example:
     *      auto what_we_have = std::make_tuple(42, 80, 'a', .1, "other_stuff", 79, .4);
     *      using what_we_need_t = std::tuple<char, double, int>;
     *      what_we_need_t expected {'a', .1, 42};
     *      auto actual = permute_to<what_we_need_t>(what_we_have);
     *      EXPECT_EQ(actual, expected);
     *
     * \tparam Res The type of resulting sequence
     * \tparam Res The type of input sequence
     *
     * \param src The input sequence
     * \return The permuted sequence
     */

    template <typename Res, typename Src>
    Res permute_to(Src &&src) {
        return impl_::permute_to_impl<Res>{}(std::forward<Src>(src));
    }
} // namespace gridtools

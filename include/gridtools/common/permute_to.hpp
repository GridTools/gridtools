/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <type_traits>
#include <utility>

#include "../meta/st_position.hpp"
#include "../meta/type_traits.hpp"
#include "../meta/utility.hpp"
#include "tuple_util.hpp"

namespace gridtools {
    namespace impl_ {
        template <typename Res>
        struct permute_to_impl;

        template <template <typename...> class Res, typename... Elems>
        struct permute_to_impl<Res<Elems...>> {
            template <typename Src>
            Res<Elems...> operator()(Src &&src) {
                using src_t = decay_t<Src>;
                return Res<Elems...>{
                    tuple_util::get<meta::st_position<src_t, Elems>::value>(wstd::forward<Src>(src))...};
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
        return impl_::permute_to_impl<Res>{}(wstd::forward<Src>(src));
    }
} // namespace gridtools

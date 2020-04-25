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

#include <cstddef>

#include <boost/preprocessor/punctuation/remove_parens.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>

#define GT_INTERNAL_DECLARE_EXPANDABLE_ICO_TMP(r, type_location, name)                                 \
    constexpr ::gridtools::stencil::expandable<::gridtools::stencil::icosahedral::tmp_arg<__COUNTER__, \
        BOOST_PP_REMOVE_PARENS(BOOST_PP_TUPLE_ELEM(2, 1, type_location)),                              \
        BOOST_PP_REMOVE_PARENS(BOOST_PP_TUPLE_ELEM(2, 0, type_location))>>                             \
        name = {};

#define GT_DECLARE_EXPANDABLE_ICO_TMP(type, location, ...)                                               \
    BOOST_PP_SEQ_FOR_EACH(                                                                               \
        GT_INTERNAL_DECLARE_EXPANDABLE_ICO_TMP, (type, location), BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)) \
    static_assert(1, "")

#define GT_INTERNAL_DECLARE_ICO_TMP(r, type_location, name)               \
    constexpr ::gridtools::stencil::icosahedral::tmp_arg<__COUNTER__,     \
        BOOST_PP_REMOVE_PARENS(BOOST_PP_TUPLE_ELEM(2, 1, type_location)), \
        BOOST_PP_REMOVE_PARENS(BOOST_PP_TUPLE_ELEM(2, 0, type_location))> \
        name = {};

#define GT_DECLARE_ICO_TMP(type, location, ...)                                                                 \
    BOOST_PP_SEQ_FOR_EACH(GT_INTERNAL_DECLARE_ICO_TMP, (type, location), BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)) \
    static_assert(1, "")

namespace gridtools {
    namespace stencil {
        namespace icosahedral {
            template <size_t, class NumColors, class Data>
            struct tmp_arg {
                using data_t = Data;
                using num_colors_t = NumColors;
                using tmp_tag = std::true_type;
            };
        } // namespace icosahedral
    }     // namespace stencil
} // namespace gridtools

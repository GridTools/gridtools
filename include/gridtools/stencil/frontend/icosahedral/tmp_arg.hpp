/*
 * GridTools
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <cstddef>
#include <type_traits>

#include <gridtools/preprocessor/punctuation/remove_parens.hpp>
#include <gridtools/preprocessor/seq/for_each.hpp>
#include <gridtools/preprocessor/tuple/elem.hpp>
#include <gridtools/preprocessor/variadic/to_seq.hpp>

#define GT_INTERNAL_DECLARE_EXPANDABLE_ICO_TMP(r, type_location, name)                                 \
    constexpr ::gridtools::stencil::expandable<::gridtools::stencil::icosahedral::tmp_arg<__COUNTER__, \
        GT_PP_REMOVE_PARENS(GT_PP_TUPLE_ELEM(2, 1, type_location)),                              \
        GT_PP_REMOVE_PARENS(GT_PP_TUPLE_ELEM(2, 0, type_location))>>                             \
        name = {};

#define GT_DECLARE_EXPANDABLE_ICO_TMP(type, location, ...)                                               \
    GT_PP_SEQ_FOR_EACH(                                                                               \
        GT_INTERNAL_DECLARE_EXPANDABLE_ICO_TMP, (type, location), GT_PP_VARIADIC_TO_SEQ(__VA_ARGS__)) \
    static_assert(1)

#define GT_INTERNAL_DECLARE_ICO_TMP(r, type_location, name)               \
    constexpr ::gridtools::stencil::icosahedral::tmp_arg<__COUNTER__,     \
        GT_PP_REMOVE_PARENS(GT_PP_TUPLE_ELEM(2, 1, type_location)), \
        GT_PP_REMOVE_PARENS(GT_PP_TUPLE_ELEM(2, 0, type_location))> \
        name = {};

#define GT_DECLARE_ICO_TMP(type, location, ...)                                                                 \
    GT_PP_SEQ_FOR_EACH(GT_INTERNAL_DECLARE_ICO_TMP, (type, location), GT_PP_VARIADIC_TO_SEQ(__VA_ARGS__)) \
    static_assert(1)

namespace gridtools {
    namespace stencil {
        namespace icosahedral {
            template <size_t I, class NumColors, class Data>
            struct tmp_arg : std::integral_constant<size_t, I> {
                using data_t = Data;
                using num_colors_t = NumColors;
                using tmp_tag = std::true_type;
            };
        } // namespace icosahedral
    }     // namespace stencil
} // namespace gridtools

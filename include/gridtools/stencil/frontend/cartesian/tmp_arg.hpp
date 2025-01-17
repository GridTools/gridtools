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
#include <gridtools/preprocessor/variadic/to_seq.hpp>

#include "../../../common/integral_constant.hpp"

#define GT_INTERNAL_DECLARE_TMP(r, type, name) \
    constexpr ::gridtools::stencil::cartesian::tmp_arg<__COUNTER__, GT_PP_REMOVE_PARENS(type)> name = {};

#define GT_DECLARE_TMP(type, ...)                                                               \
    GT_PP_SEQ_FOR_EACH(GT_INTERNAL_DECLARE_TMP, type, GT_PP_VARIADIC_TO_SEQ(__VA_ARGS__)) \
    static_assert(1)

#define GT_INTERNAL_DECLARE_EXPANDABLE_TMP(r, type, name)                                    \
    constexpr ::gridtools::stencil::expandable<                                              \
        ::gridtools::stencil::cartesian::tmp_arg<__COUNTER__, GT_PP_REMOVE_PARENS(type)>> \
        name = {};

#define GT_DECLARE_EXPANDABLE_TMP(type, ...)                                                               \
    GT_PP_SEQ_FOR_EACH(GT_INTERNAL_DECLARE_EXPANDABLE_TMP, type, GT_PP_VARIADIC_TO_SEQ(__VA_ARGS__)) \
    static_assert(1)

namespace gridtools {
    namespace stencil {
        namespace cartesian {
            template <size_t I, class Data>
            struct tmp_arg : std::integral_constant<size_t, I> {
                using data_t = Data;
                using num_colors_t = integral_constant<int_t, 1>;
                using tmp_tag = std::true_type;
            };
        } // namespace cartesian
    }     // namespace stencil
} // namespace gridtools

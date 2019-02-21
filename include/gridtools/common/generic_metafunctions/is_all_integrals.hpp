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

#include "../../meta/macros.hpp"
#include "../../meta/type_traits.hpp"

namespace gridtools {

    /** \ingroup common
        @{
        \ingroup allmeta
        @{
        \defgroup variadic Variadic Pack Utilities
        @{
    */

    /* check if all given types are integral types */
    template <typename... IntTypes>
    GT_META_DEFINE_ALIAS(is_all_integral, conjunction, std::is_integral<IntTypes>...);

    template <typename T>
    GT_META_DEFINE_ALIAS(is_integral_or_enum, bool_constant, std::is_integral<T>::value || std::is_enum<T>::value);

    /* check if all given types are integral types or enums */
    template <typename... IntTypes>
    GT_META_DEFINE_ALIAS(is_all_integral_or_enum, conjunction, is_integral_or_enum<IntTypes>...);
    /** @} */
    /** @} */
    /** @} */
} // namespace gridtools

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

#include "../meta/utility.hpp"
#include "./array.hpp"
#include "./generic_metafunctions/accumulate.hpp"

namespace gridtools {
    /** \addtogroup common
        @{
     */

    /** \addtogroup array
        @{
    */

    namespace _impl {
        template <typename T, typename U, size_t D, size_t... Is>
        GT_FUNCTION GT_CONSTEXPR auto dot_impl(
            array<T, D> const &a, array<U, D> const &b, meta::integer_sequence<size_t, Is...>)
            -> decltype(accumulate(plus_functor{}, (a[Is] * b[Is])...)) {
            return accumulate(plus_functor{}, (a[Is] * b[Is])...);
        }
    } // namespace _impl

    /**
     * @brief dot product for gridtools::array (enabled for all arithmetic types)
     *
     * @tparam T Element type of first array.
     * @tparam U Element type of second array.
     * @tparam D Array size.
     *
     * @param a First array.
     * @param b Second array.
     *
     * \return Value corresponding to the first array value type
     */
    template <typename T,
        typename U,
        size_t D,
        typename std::enable_if<std::is_arithmetic<T>::value and std::is_arithmetic<U>::value, T>::type = 0>
    GT_FUNCTION GT_CONSTEXPR T array_dot_product(array<T, D> const &a, array<U, D> const &b) {
        return _impl::dot_impl(a, b, meta::make_integer_sequence<size_t, D>{});
    }

    /** @} */
    /** @} */

} // namespace gridtools

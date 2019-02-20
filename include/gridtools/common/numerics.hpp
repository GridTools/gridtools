/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "defs.hpp"
#include "host_device.hpp"

namespace gridtools {
    namespace _impl {
        /** \ingroup common
            @{
            \defgroup numerics Compile-Time Numerics
            @{
        */

        /** @brief Compute 3^I at compile time
            \tparam I Exponent
        */
        template <uint_t I>
        struct static_pow3;

        template <>
        struct static_pow3<0> {
            static const int value = 1;
        };

        template <>
        struct static_pow3<1> {
            static const int value = 3;
        };

        template <uint_t I>
        struct static_pow3 {
            static const int value = 3 * static_pow3<I - 1>::value;
        };
        /** @} */
        /** @} */
    } // namespace _impl
} // namespace gridtools

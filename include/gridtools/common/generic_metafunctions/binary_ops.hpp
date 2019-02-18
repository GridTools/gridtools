/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#include "../host_device.hpp"

namespace gridtools {

    /**@brief operation to be used inside the accumulator*/
    struct logical_and {
        GT_FUNCTION
        constexpr logical_and() {}
        template <typename T>
        GT_FUNCTION constexpr T operator()(const T &x, const T &y) const {
            return x && y;
        }
    };

    /**@brief operation to be used inside the accumulator*/
    struct logical_or {
        GT_FUNCTION
        constexpr logical_or() {}
        template <typename T>
        GT_FUNCTION constexpr T operator()(const T &x, const T &y) const {
            return x || y;
        }
    };

    /**@brief binary operator functor that checks if two types passed fulfill the == operator*/
    struct equal {
        GT_FUNCTION
        constexpr equal() {}

        template <typename T>
        GT_FUNCTION constexpr bool operator()(const T &x, const T &y) const {
            return x == y;
        }
    };
} // namespace gridtools

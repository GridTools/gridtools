/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <tuple>
#include <utility>

#include "../common/integral_constant.hpp"

namespace gridtools::fn {
    using i_t = integral_constant<int, 0>;
    using j_t = integral_constant<int, 1>;
    using k_t = integral_constant<int, 2>;

    inline constexpr i_t i = {};
    inline constexpr j_t j = {};
    inline constexpr k_t k = {};

    template <class Sizes, class Offsets = std::tuple<>, class Vertical = k_t>
    struct cartesian {
        Sizes sizes;
        Offsets offsets;
        constexpr cartesian(Sizes sizes, Offsets offsets = {}, Vertical = {}) : sizes(sizes), offsets(offsets) {}
    };

} // namespace gridtools::fn

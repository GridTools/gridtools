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
    using hor_t = integral_constant<int, 0>;
    inline constexpr hor_t hor = {};

    using vert_t = integral_constant<int, 1>;
    inline constexpr vert_t vert = {};

    template <class Sizes, class Offsets = std::tuple<>, class Horizontal = hor_t, class Vertical = vert_t>
    struct unstructured {
        Sizes sizes;
        Offsets offsets;
        constexpr unstructured(Sizes sizes, Offsets offsets = {}, Horizontal = {}, Vertical = {})
            : sizes(sizes), offsets(std::move(offsets)) {}
    };
} // namespace gridtools::fn

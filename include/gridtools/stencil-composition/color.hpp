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

#include "../common/defs.hpp"

namespace gridtools {
    template <uint_t c>
    struct color_type {
        using color_t = static_uint<c>;
    };

    struct nocolor {
        using color_t = void;
    };

    template <typename T>
    struct is_color_type : std::false_type {};

    template <uint_t c>
    struct is_color_type<color_type<c>> : std::true_type {};

    template <>
    struct is_color_type<nocolor> : std::true_type {};
}; // namespace gridtools

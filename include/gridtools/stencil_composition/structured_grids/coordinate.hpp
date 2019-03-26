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

#include "../../common/defs.hpp"

namespace gridtools {
    template <class /*Target*/>
    struct coord_i : std::integral_constant<size_t, 0> {};

    template <class /*Target*/>
    struct coord_j : std::integral_constant<size_t, 1> {};

    template <class /*Target*/>
    struct coord_k : std::integral_constant<size_t, 2> {};
} // namespace gridtools

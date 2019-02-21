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
#include "../backend_ids.hpp"

namespace gridtools {
    template <class>
    struct coord_i;

    template <class BackendId, class StrategyId>
    struct coord_i<backend_ids<BackendId, grid_type::icosahedral, StrategyId>> : std::integral_constant<size_t, 0> {};

    template <class>
    struct coord_j;

    template <class BackendId, class StrategyId>
    struct coord_j<backend_ids<BackendId, grid_type::icosahedral, StrategyId>> : std::integral_constant<size_t, 2> {};

    template <class>
    struct coord_k;

    template <class BackendId, class StrategyId>
    struct coord_k<backend_ids<BackendId, grid_type::icosahedral, StrategyId>> : std::integral_constant<size_t, 3> {};
} // namespace gridtools

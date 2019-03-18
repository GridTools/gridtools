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

    template <class Platform, class Strategy>
    struct coord_i<backend_ids<Platform, Strategy>> : std::integral_constant<size_t, 0> {};

    template <class>
    struct coord_j;

    template <class Platform, class Strategy>
    struct coord_j<backend_ids<Platform, Strategy>> : std::integral_constant<size_t, 1> {};

    template <class>
    struct coord_k;

    template <class Platform, class Strategy>
    struct coord_k<backend_ids<Platform, Strategy>> : std::integral_constant<size_t, 2> {};
} // namespace gridtools

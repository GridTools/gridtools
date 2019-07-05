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

#include <tuple>

#include "../../common/defs.hpp"
#include "esf.hpp"

namespace gridtools {
    template <template <uint_t> class Functor, typename Grid, typename LocationType, typename... Args>
    constexpr std::tuple<esf_descriptor<Functor, Grid, LocationType, std::tuple<Args...>>> make_stage(Args...) {
        return {};
    }
} // namespace gridtools

/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <tuple>

#include "../../common/defs.hpp"
#include "./esf.hpp"

namespace gridtools {

    template <template <uint_t> class Functor, typename Grid, typename LocationType, typename... Args>
    esf_descriptor<Functor, Grid, LocationType, nocolor, std::tuple<Args...>> make_stage(Args...) {
        return {};
    }
} // namespace gridtools

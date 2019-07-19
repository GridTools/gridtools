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
#include <utility>

#include "../../meta.hpp"
#include "../computation_facade.hpp"
#include "../mss.hpp"
#include "entry_point.hpp"
#include "expand_factor.hpp"

namespace gridtools {

#ifndef NDEBUG
#define GT_POSITIONAL_WHEN_DEBUGGING true
#else
#define GT_POSITIONAL_WHEN_DEBUGGING false
#endif

    template <class Backend, bool IsStateful = GT_POSITIONAL_WHEN_DEBUGGING, size_t N, class... Args>
    auto make_expandable_computation(expand_factor<N>, Args... args) {
        using msses_t = meta::filter<is_mss_descriptor, std::tuple<Args...>>;
        using expandable_entry_point =
            expandable_entry_point_f<expand_factor<N>, Backend, bool_constant<IsStateful>, msses_t>;
        return make_computation_facade<Backend, expandable_entry_point>(std::move(args)...);
    }

#undef GT_POSITIONAL_WHEN_DEBUGGING

    // user protection only, catch the case where no backend is specified
    template <class... Args>
    void make_expandable_computation(Args &&...) {
        GT_STATIC_ASSERT((sizeof...(Args), false), "No backend was specified on a call to make_computation");
    }

    template <class Backend, class Grid, size_t N, class... Args>
    auto make_expandable_positional_computation(expand_factor<N> factor, Grid const &grid, Args... args) {
        return make_expandable_computation<Backend, true, Grid>(factor, grid, std::move(args)...);
    }
} // namespace gridtools

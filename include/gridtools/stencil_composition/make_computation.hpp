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

#include "../common/defs.hpp"
#include "../meta.hpp"
#include "backend.hpp"
#include "computation_facade.hpp"
#include "mss.hpp"

namespace gridtools {

#ifndef NDEBUG
#define GT_POSITIONAL_WHEN_DEBUGGING true
#else
#define GT_POSITIONAL_WHEN_DEBUGGING false
#endif

    template <class Backend, bool IsStateful = GT_POSITIONAL_WHEN_DEBUGGING, class... Args>
    auto make_computation(Args... args) {
        using msses_t = meta::filter<is_mss_descriptor, std::tuple<Args...>>;
        using entry_point_t = backend_entry_point_f<Backend, bool_constant<IsStateful>, msses_t>;
        return make_computation_facade<Backend, entry_point_t>(std::move(args)...);
    }

#undef GT_POSITIONAL_WHEN_DEBUGGING

    // user protection only, catch the case where no backend is specified
    template <class... Args>
    void make_computation(Args &&...) {
        GT_STATIC_ASSERT((sizeof...(Args), false), "No backend was specified on a call to make_computation");
    }

    template <class Backend, class... Args>
    auto make_positional_computation(Args &&... args) {
        return make_computation<Backend, true>(std::forward<Args>(args)...);
    }
} // namespace gridtools

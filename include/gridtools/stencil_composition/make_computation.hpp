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

#include <utility>

#include "../common/defs.hpp"
#include "../meta.hpp"
#include "backend.hpp"
#include "computation_facade.hpp"
#include "mss.hpp"

namespace gridtools {
    template <class Backend, class... Args>
    auto make_computation_old(Args... args) {
        using msses_t = meta::filter<is_mss_descriptor, meta::list<Args...>>;
        using entry_point_t = backend_entry_point_f<Backend, msses_t>;
        return make_computation_facade<Backend, entry_point_t>(std::move(args)...);
    }

    // user protection only, catch the case where no backend is specified
    template <class... Args>
    void make_computation_old(Args &&...) {
        static_assert((sizeof...(Args), false), "No backend was specified on a call to make_computation");
    }

    template <class Backend, class... Args>
    void compute(Args &&... args) {
        make_computation_old<Backend>(std::forward<Args>(args)...).run();
    }
} // namespace gridtools

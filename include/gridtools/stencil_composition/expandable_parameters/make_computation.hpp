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

#include "../../meta.hpp"
#include "../computation_facade.hpp"
#include "../mss.hpp"
#include "entry_point.hpp"
#include "expand_factor.hpp"

namespace gridtools {
    template <class Backend, size_t N, class... Args>
    auto make_expandable_computation_old(expand_factor<N>, Args... args) {
        using msses_t = meta::filter<is_mss_descriptor, meta::list<Args...>>;
        using expandable_entry_point = expandable_entry_point_f<expand_factor<N>, Backend, msses_t>;
        return make_computation_facade<Backend, expandable_entry_point>(std::move(args)...);
    }

    // user protection only, catch the case where no backend is specified
    template <class... Args>
    void make_expandable_computation_old(Args &&...) {
        static_assert((sizeof...(Args), false), "No backend was specified on a call to make_computation");
    }

    template <class Backend, class... Args>
    void expandable_compute(Args &&... args) {
        make_expandable_computation_old<Backend>(std::forward<Args>(args)...).run();
    }

} // namespace gridtools

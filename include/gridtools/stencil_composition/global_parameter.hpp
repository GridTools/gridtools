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

#include <cassert>

#include "../storage/storage_facility.hpp"

namespace gridtools {
    template <class Target, class T>
    using global_parameter = typename storage_traits<Target>::template data_store_t<T,
        typename storage_traits<Target>::template special_storage_info_t<0, selector<0>, zero_halo<1>>>;

    template <class Target, class T>
    global_parameter<Target, T> make_global_parameter(T const &value) {
        typename global_parameter<Target, T>::storage_info_t si(1);
        global_parameter<Target, T> ds(si);
        make_host_view(ds)(0) = value;
        ds.sync();
        return ds;
    }

    template <class GlobalParameter, class T>
    void update_global_parameter(GlobalParameter &gp, T const &value) {
        gp.sync();
        auto view = make_host_view(gp);
        assert(check_consistency(gp, view) && "Cannot create a valid view to a global parameter. Properly synced?");
        view(0) = value;
        gp.sync();
    }
} // namespace gridtools

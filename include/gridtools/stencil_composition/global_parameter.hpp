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
    /**
        Method to retrieve a global parameter
     */
    template <class Target, class T>
    static typename storage_traits<Target>::template data_store_t<T,
        typename storage_traits<Target>::template special_storage_info_t<0, selector<0u>, zero_halo<1>>>
    make_global_parameter(T const &value) {
        typename storage_traits<Target>::template special_storage_info_t<0, selector<0u>, zero_halo<1>> si(1);
        typename storage_traits<Target>::template data_store_t<T, decltype(si)> ds(si);
        make_host_view(ds)(0) = value;
        ds.sync();
        return ds;
    }

    /**
        Method to update a global parameter
     */
    template <class GlobalParameter, class T>
    static void update_global_parameter(GlobalParameter &gp, T const &value) {
        gp.sync();
        auto view = make_host_view(gp);
        assert(check_consistency(gp, view) && "Cannot create a valid view to a global parameter. Properly synced?");
        view(0) = value;
        gp.sync();
    }
} // namespace gridtools

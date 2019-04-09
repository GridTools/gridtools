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
    template <class Backend, class T>
    using global_parameter = typename storage_traits<Backend>::template data_store_t<T,
        typename storage_traits<Backend>::template special_storage_info_t<0, selector<0>, halo<0>>>;

    template <class Backend, class T>
    global_parameter<Backend, T> make_global_parameter(T const &value) {
        return {{1}, value};
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

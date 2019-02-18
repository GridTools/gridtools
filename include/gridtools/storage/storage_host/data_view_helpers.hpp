/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <assert.h>

#include "../../common/gt_assert.hpp"
#include "../../meta/type_traits.hpp"
#include "../data_store.hpp"
#include "../data_view.hpp"
#include "host_storage.hpp"

namespace gridtools {

    /** \ingroup storage
     * @{
     */

    /**
     * @brief function used to create views to data stores (read-write/read-only).
     * @tparam AccessMode access mode information (default is read-write).
     * @param ds data store
     * @return a host view to the given data store.
     */
    template <access_mode AccessMode = access_mode::read_write,
        typename DataStore,
        typename DecayedDS = decay_t<DataStore>>
    enable_if_t<is_host_storage<typename DecayedDS::storage_t>::value &&
                    is_storage_info<typename DecayedDS::storage_info_t>::value && is_data_store<DecayedDS>::value,
        data_view<DataStore, AccessMode>>
    make_host_view(DataStore const &ds) {
        return ds.valid() ? data_view<DecayedDS, AccessMode>(ds.get_storage_ptr()->get_cpu_ptr(),
                                ds.get_storage_info_ptr().get(),
                                ds.get_storage_ptr()->get_state_machine_ptr(),
                                false)
                          : data_view<DecayedDS, AccessMode>();
    }

    /**
     * @brief Create a view to the target (host view for host storage, device view for cuda storage)
     * @tparam AccessMode access mode information (default is read-write).
     * @param ds data store
     * @return a host view to the given data store.
     */
    template <access_mode AccessMode = access_mode::read_write,
        typename DataStore,
        typename DecayedDS = decay_t<DataStore>>
    enable_if_t<is_host_storage<typename DecayedDS::storage_t>::value &&
                    is_storage_info<typename DecayedDS::storage_info_t>::value && is_data_store<DecayedDS>::value,
        data_view<DataStore, AccessMode>>
    make_target_view(DataStore const &ds) {
        return make_host_view<AccessMode>(ds);
    }

    /**
     * @brief function that can be used to check if a view is in a consistent state
     * @param ds data store
     * @param dv data view
     * @return true if the given view is in a valid state and can be used safely.
     */
    template <typename DataStore,
        typename DataView,
        typename DecayedDS = decay_t<DataStore>,
        typename DecayedDV = decay_t<DataView>>
    enable_if_t<is_host_storage<typename DecayedDS::storage_t>::value &&
                    is_storage_info<typename DecayedDS::storage_info_t>::value && is_data_store<DecayedDS>::value,
        bool>
    check_consistency(DataStore const &ds, DataView const &dv) {
        GT_STATIC_ASSERT(is_data_view<DecayedDV>::value, GT_INTERNAL_ERROR_MSG("Passed type is no data_view type"));
        return ds.valid() && advanced::get_raw_pointer_of(dv) == ds.get_storage_ptr()->get_cpu_ptr() &&
               advanced::storage_info_raw_ptr(dv) && ds.get_storage_info_ptr().get();
    }

    /**
     * @}
     */
} // namespace gridtools

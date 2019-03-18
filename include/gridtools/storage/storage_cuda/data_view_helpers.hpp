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

#include <assert.h>

#include "../../common/gt_assert.hpp"
#include "../../meta/type_traits.hpp"
#include "../data_store.hpp"
#include "../data_view.hpp"
#include "cuda_storage.hpp"
#include "cuda_storage_info.hpp"

namespace gridtools {
    /** \ingroup storage
     * @{
     */

    /**
     * @brief function used to create host views to data stores (read-write/read-only).
     * @tparam AccessMode access mode information (default is read-write).
     * @param ds data store
     * @return a host view to the given data store.
     */
    template <access_mode AccessMode = access_mode::read_write,
        typename CudaDataStore,
        typename Res = data_view<CudaDataStore, AccessMode>>
    enable_if_t<is_cuda_storage<typename CudaDataStore::storage_t>::value &&
                    is_storage_info<typename CudaDataStore::storage_info_t>::value &&
                    is_data_store<CudaDataStore>::value,
        Res>
    make_host_view(CudaDataStore const &ds) {
        if (!ds.valid())
            return {};

        if (AccessMode != access_mode::read_only) {
            GT_ASSERT_OR_THROW(!ds.get_storage_ptr()->get_state_machine_ptr()->m_hnu,
                "There is already an active read-write "
                "device view. Synchronization is needed "
                "before constructing the view.");
            ds.get_storage_ptr()->get_state_machine_ptr()->m_dnu = true;
        }
        return {ds.get_storage_ptr()->get_cpu_ptr(),
            ds.get_storage_info_ptr().get(),
            ds.get_storage_ptr()->get_state_machine_ptr(),
            false};
    }

    /**
     * @brief function used to create device views to data stores (read-write/read-only).
     * @tparam AccessMode access mode information (default is read-write).
     * @param ds data store
     * @return a device view to the given data store.
     */
    template <access_mode AccessMode = access_mode::read_write,
        typename CudaDataStore,
        typename Res = data_view<CudaDataStore, AccessMode>>
    enable_if_t<is_cuda_storage<typename CudaDataStore::storage_t>::value &&
                    is_storage_info<typename CudaDataStore::storage_info_t>::value &&
                    is_data_store<CudaDataStore>::value,
        Res>
    make_device_view(CudaDataStore const &ds) {
        if (!ds.valid())
            return {};

        if (AccessMode != access_mode::read_only) {
            GT_ASSERT_OR_THROW(!ds.get_storage_ptr()->get_state_machine_ptr()->m_dnu,
                "There is already an active read-write "
                "host view. Synchronization is needed "
                "before constructing the view.");
            ds.get_storage_ptr()->get_state_machine_ptr()->m_hnu = true;
        }
        return {ds.get_storage_ptr()->get_gpu_ptr(),
            get_gpu_storage_info_ptr(*ds.get_storage_info_ptr()),
            ds.get_storage_ptr()->get_state_machine_ptr(),
            true};
    }

    /**
     * @brief Create a view to the target (host view for host storage, device view for cuda storage)
     * @tparam AccessMode access mode information (default is read-write).
     * @param ds data store
     * @return a device view to the given data store.
     */
    template <access_mode AccessMode = access_mode::read_write,
        typename CudaDataStore,
        typename Res = data_view<CudaDataStore, AccessMode>>
    enable_if_t<is_cuda_storage<typename CudaDataStore::storage_t>::value &&
                    is_storage_info<typename CudaDataStore::storage_info_t>::value &&
                    is_data_store<CudaDataStore>::value,
        Res>
    make_target_view(CudaDataStore const &ds) {
        return make_device_view<AccessMode>(ds);
    }

    /**
     * @brief function that can be used to check if a view is in a consistent state
     * @param d data store
     * @param v data view
     * @return true if the given view is in a valid state and can be used safely.
     */
    template <typename DataStore, typename DataView>
    enable_if_t<is_cuda_storage<typename DataStore::storage_t>::value &&
                    is_storage_info<typename DataStore::storage_info_t>::value && is_data_store<DataStore>::value,
        bool>
    check_consistency(DataStore const &d, DataView const &v) {
        GT_STATIC_ASSERT(is_data_view<DataView>::value, "Passed type is no data_view type");
        // if the storage is not valid return false
        if (!d.valid())
            return false;
        // if ptrs do not match anymore return false
        if (advanced::get_raw_pointer_of(v) != d.get_storage_ptr()->get_gpu_ptr() &&
            advanced::get_raw_pointer_of(v) != d.get_storage_ptr()->get_cpu_ptr())
            return false;
        // check if we have a device view
        bool device_view = advanced::get_raw_pointer_of(v) != d.get_storage_ptr()->get_cpu_ptr();
        // read-only? if yes, take early exit
        if (DataView::mode == access_mode::read_only)
            return device_view ? !d.get_storage_ptr()->get_state_machine_ptr()->m_dnu
                               : !d.get_storage_ptr()->get_state_machine_ptr()->m_hnu;
        else
            // get storage state
            return device_view ? d.get_storage_ptr()->get_state_machine_ptr()->m_hnu &&
                                     !d.get_storage_ptr()->get_state_machine_ptr()->m_dnu
                               : !d.get_storage_ptr()->get_state_machine_ptr()->m_hnu &&
                                     d.get_storage_ptr()->get_state_machine_ptr()->m_dnu;
    }

    /**
     * @}
     */
} // namespace gridtools

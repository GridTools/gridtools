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

#include <boost/mpl/if.hpp>
#include <boost/noncopyable.hpp>
#include <boost/type_traits.hpp>

#include "../../common/error.hpp"
#include "definitions.hpp"

namespace gridtools {

    /** \ingroup storage
     * @{
     */

    struct state_machine;

    /*
     * @brief The storage interface. This struct defines a set of methods that should be
     * available in the actual storage class.
     * @tparam Derived The actual storage type (e.g., cuda_storage or host_storage)
     */
    template <typename Derived>
    struct storage_interface : boost::noncopyable {
        /*
         * @brief clone_to_device method. Clones data to the device.
         */
        void clone_to_device() { static_cast<Derived *>(this)->clone_to_device_impl(); }

        /*
         * @brief clone_from_device method. Clones data from the device.
         */
        void clone_from_device() { static_cast<Derived *>(this)->clone_from_device_impl(); }

        /*
         * @brief sync method. Synchronize the data. This means either cloning from or to device
         * or not doing anything (e.g., host_storage, or cuda_storage without an active ReadWrite view).
         */
        void sync() { static_cast<Derived *>(this)->sync_impl(); }

        /*
         * @brief This method retrieves if the state machine is in a "device needs update" state.
         * @return information if device needs update
         */
        bool device_needs_update() const { return static_cast<Derived const *>(this)->device_needs_update_impl(); }

        /*
         * @brief This method retrieves if the state machine is in a "host needs update" state.
         * @return information if host needs update
         */
        bool host_needs_update() const { return static_cast<Derived const *>(this)->host_needs_update_impl(); }

        /*
         * @brief This method sets the state machine to a "host needs update" state. This means that
         * previously created device write views will appear as valid views that can be used.
         */
        void reactivate_device_write_views() { static_cast<Derived *>(this)->reactivate_device_write_views_impl(); }

        /*
         * @brief This method sets the state machine to a "device needs update" state. This means that
         * previously created host write views will appear as valid views that can be used.
         */
        void reactivate_host_write_views() { static_cast<Derived *>(this)->reactivate_host_write_views_impl(); }

        /*
         * @brief This method sets the state machine to a "device needs update" state. This means that
         * previously created host write views will appear as valid views that can be used.
         * @return pointer to the state machine
         */
        state_machine *get_state_machine_ptr() { return static_cast<Derived *>(this)->get_state_machine_ptr_impl(); }

        /*
         * @brief This method swaps the data of two storages.
         */
        void swap(storage_interface &other) { static_cast<Derived *>(this)->swap_impl(static_cast<Derived &>(other)); }

        /*
         * @brief This method retrieves all pointers that are contained in the storage (in case of host_storage
         * only one pointer, in case of cuda_storage two pointers).
         * @return struct that contains the pointer(s)
         */
        template <typename T>
        T get_ptrs() const {
            return static_cast<Derived const *>(this)->get_ptrs_impl();
        }

        /*
         * @brief This method returns information about validity of the storage (e.g., no nullptrs, etc.).
         * @return true if the storage is valid, false otherwise
         */
        bool valid() const { return static_cast<Derived const *>(this)->valid_impl(); }
    };

    template <typename T>
    struct is_storage : boost::is_base_of<storage_interface<T>, T> {};

    /**
     * @}
     */
} // namespace gridtools

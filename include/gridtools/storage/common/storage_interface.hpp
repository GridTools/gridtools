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

#include <type_traits>

#include "definitions.hpp"

namespace gridtools {

    /** \ingroup storage
     * @{
     */

    /*
     * @brief The storage interface. This struct defines a set of methods that should be
     * available in the actual storage class.
     * @tparam Derived The actual storage type (e.g., cuda_storage or host_storage)
     */
    template <typename Derived>
    class storage_interface {
        Derived &derived() { return static_cast<Derived &>(*this); }
        Derived const &derived() const { return static_cast<Derived const &>(*this); }

        void clone_to_device_impl() {}
        void clone_from_device_impl() {}
        void sync_impl() {}
        bool device_needs_update_impl() const { return false; };
        bool host_needs_update_impl() const { return false; };
        void reactivate_target_write_views_impl() {}
        void reactivate_host_write_views_impl() {}
        auto get_target_ptr_impl() const { return derived().get_cpu_ptr_impl(); }

      public:
        /*
         * @brief clone_to_device method. Clones data to the device.
         */
        void clone_to_device() { derived().clone_to_device_impl(); }

        /*
         * @brief clone_from_device method. Clones data from the device.
         */
        void clone_from_device() { derived().clone_from_device_impl(); }

        /*
         * @brief sync method. Synchronize the data. This means either cloning from or to device
         * or not doing anything (e.g., host_storage, or cuda_storage without an active ReadWrite view).
         */
        void sync() { derived().sync_impl(); }

        /*
         * @brief This method retrieves if the state machine is in a "device needs update" state.
         * @return information if device needs update
         */
        bool device_needs_update() const { return derived().device_needs_update_impl(); }

        /*
         * @brief This method retrieves if the state machine is in a "host needs update" state.
         * @return information if host needs update
         */
        bool host_needs_update() const { return derived().host_needs_update_impl(); }

        /*
         * @brief This method sets the state machine to a "host needs update" state. This means that
         * previously created device write views will appear as valid views that can be used.
         */
        void reactivate_target_write_views() { derived().reactivate_target_write_views_impl(); }

        /*
         * @brief This method sets the state machine to a "device needs update" state. This means that
         * previously created host write views will appear as valid views that can be used.
         */
        void reactivate_host_write_views() { derived().reactivate_host_write_views_impl(); }

        auto get_cpu_ptr() const { return derived().get_cpu_ptr_impl(); }

        auto get_target_ptr() const { return derived().get_target_ptr_impl(); }
    };

    template <typename T>
    struct is_storage : std::is_base_of<storage_interface<T>, T> {};

    /**
     * @}
     */
} // namespace gridtools

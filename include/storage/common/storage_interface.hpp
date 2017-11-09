/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/

#pragma once

#include <boost/mpl/if.hpp>
#include <boost/noncopyable.hpp>
#include <boost/type_traits.hpp>

#include "../../common/error.hpp"
#include "definitions.hpp"

namespace gridtools {

    struct state_machine;

    /*
     * @brief The storage interface. This struct defines a set of methods that should be
     * available in the actual storage class.
     * @tparam Derived The actual storage type (e.g., cuda_storage or host_storage)
     */
    template < typename Derived >
    struct storage_interface : boost::noncopyable {
        /*
         * @brief clone_to_device method. Clones data to the device.
         */
        void clone_to_device() { static_cast< Derived * >(this)->clone_to_device_impl(); }

        /*
         * @brief clone_from_device method. Clones data from the device.
         */
        void clone_from_device() { static_cast< Derived * >(this)->clone_from_device_impl(); }

        /*
         * @brief sync method. Synchronize the data. This means either cloning from or to device
         * or not doing anything (e.g., host_storage, or cuda_storage without an active ReadWrite view).
         */
        void sync() { static_cast< Derived * >(this)->sync_impl(); }

        /*
         * @brief This method retrieves if the state machine is in a "device needs update" state.
         * @return information if device needs update
         */
        bool device_needs_update() const { return static_cast< Derived const * >(this)->device_needs_update_impl(); }

        /*
         * @brief This method retrieves if the state machine is in a "host needs update" state.
         * @return information if host needs update
         */
        bool host_needs_update() const { return static_cast< Derived const * >(this)->host_needs_update_impl(); }

        /*
         * @brief This method sets the state machine to a "host needs update" state. This means that
         * previously created device write views will appear as valid views that can be used.
         */
        void reactivate_device_write_views() { static_cast< Derived * >(this)->reactivate_device_write_views_impl(); }

        /*
         * @brief This method sets the state machine to a "device needs update" state. This means that
         * previously created host write views will appear as valid views that can be used.
         */
        void reactivate_host_write_views() { static_cast< Derived * >(this)->reactivate_host_write_views_impl(); }

        /*
         * @brief This method sets the state machine to a "device needs update" state. This means that
         * previously created host write views will appear as valid views that can be used.
         * @return pointer to the state machine
         */
        state_machine *get_state_machine_ptr() { return static_cast< Derived * >(this)->get_state_machine_ptr_impl(); }

        /*
         * @brief This method swaps the data of two storages.
         */
        void swap(storage_interface &other) {
            return static_cast< Derived * >(this)->swap_impl(static_cast< Derived & >(other));
        }

        /*
         * @brief This method retrieves all pointers that are contained in the storage (in case of host_storage
         * only one pointer, in case of cuda_storage two pointers).
         * @return struct that contains the pointer(s)
         */
        template < typename T >
        T get_ptrs() const {
            return static_cast< Derived const * >(this)->get_ptrs_impl();
        }

        /*
         * @brief This method resets the pointer(s) that are contained in a storage.
         * @param ptrs struct that contains the pointer(s)
         */
        template < typename T >
        void set_ptrs(T const &ptrs) {
            static_cast< Derived * >(this)->set_ptrs_impl(ptrs);
        }

        /*
         * @brief This method returns information about validity of the storage (e.g., no nullptrs, etc.).
         * @return true if the storage is valid, false otherwise
         */
        bool valid() const { return static_cast< Derived const * >(this)->valid_impl(); }
    };

    template < typename T >
    struct is_storage : boost::is_base_of< storage_interface< T >, T > {};
}

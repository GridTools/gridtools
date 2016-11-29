/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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

#include <boost/noncopyable.hpp>
#include <boost/mpl/if.hpp>
#include <boost/type_traits.hpp>

namespace gridtools {

    struct state_machine;

    template < typename Derived >
    struct storage_interface : boost::noncopyable {
        // cloning methods
        void clone_to_device() { static_cast< Derived * >(this)->clone_to_device_impl(); }
        void clone_from_device() { static_cast< Derived * >(this)->clone_from_device_impl(); }
        void sync() { static_cast< Derived * >(this)->sync_impl(); }

        // checking the state
        bool is_on_host() const { return static_cast< Derived const * >(this)->is_on_host_impl(); }
        bool is_on_device() const { return static_cast< Derived const * >(this)->is_on_device_impl(); }
        bool device_needs_update() const { return static_cast< Derived const * >(this)->device_needs_update_impl(); }
        bool host_needs_update() const { return static_cast< Derived const * >(this)->host_needs_update_impl(); }
        void reactivate_device_write_views() { static_cast< Derived * >(this)->reactivate_device_write_views_impl(); }
        void reactivate_host_write_views() { static_cast< Derived * >(this)->reactivate_host_write_views_impl(); }
        state_machine *get_state_machine_ptr() { return static_cast< Derived * >(this)->get_state_machine_ptr_impl(); }

        // interface used for swap operations
        template < typename T >
        T get_ptrs() const {
            return static_cast< Derived const * >(this)->get_ptrs_impl();
        }

        template < typename T >
        void set_ptrs(T const &ptrs) {
            static_cast< Derived * >(this)->set_ptrs_impl(ptrs);
        }

        // interface to check validity
        bool valid() const { return static_cast< Derived const * >(this)->valid_impl(); }
    };

    template < typename T >
    struct is_storage : boost::is_base_of< storage_interface< T >, T > {};
}

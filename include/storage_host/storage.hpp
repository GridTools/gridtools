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

#include <assert.h>

#include "../common/storage_interface.hpp"
#include "../common/state_machine.hpp"

namespace gridtools {

    template < typename T >
    struct host_storage : storage_interface< host_storage< T > > {
        typedef T data_t;
        typedef T *ptrs_t;
        typedef state_machine state_machine_t;

      private:
        T *m_cpu_ptr;

      public:
        constexpr host_storage(unsigned size) : m_cpu_ptr(new T[size]) {}

        host_storage(unsigned size, T initializer) : m_cpu_ptr(new T[size]) {
            for(unsigned i=0; i<size; ++i) {
                m_cpu_ptr[i] = initializer;
            }
        }

        ~host_storage() {
            assert(m_cpu_ptr && "This would end up in a double-free.");
            delete[] m_cpu_ptr;
        }

        T *get_cpu_ptr() const {
            assert(m_cpu_ptr && "This storage has never been initialized");
            return m_cpu_ptr;
        }

        // interface used for swap operations
        ptrs_t get_ptrs_impl() const { return m_cpu_ptr; }

        void set_ptrs_impl(ptrs_t ptr) { m_cpu_ptr = ptr; }

        // interface used to check validity
        bool valid_impl() const { return m_cpu_ptr; }

        // interface compatibility
        void clone_to_device_impl(){};
        void clone_from_device_impl(){};
        void sync_impl(){};
        bool is_on_host_impl() const { return true; }
        bool is_on_device_impl() const { return true; }
        bool device_needs_update_impl() const { return false; }
        bool host_needs_update_impl() const { return false; }
        void reactivate_device_write_views_impl() {}
        void reactivate_host_write_views_impl() {}
        state_machine *get_state_machine_ptr_impl() { return nullptr; }
    };

    // simple metafunction to check if a type is a cuda storage
    template < typename T >
    struct is_host_storage : boost::mpl::false_ {};

    template < typename T >
    struct is_host_storage< host_storage< T > > : boost::mpl::true_ {};
}

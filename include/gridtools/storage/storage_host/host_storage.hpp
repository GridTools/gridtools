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
#include <assert.h>
#include <iostream>

#include "../../common/gt_assert.hpp"
#include "../common/alignment.hpp"
#include "../common/state_machine.hpp"
#include "../common/storage_interface.hpp"
#include <cstddef>
namespace gridtools {

    /** \ingroup storage
     * @{
     */

    /*
     * @brief The Host storage implementation. This class owns the pointer
     * to the data. Additionally there is a field that contains information about
     * the ownership. Instances of this class are noncopyable.
     * @tparam DataType the type of the data and the pointer respectively (e.g., float or double)
     *
     * Here we are using the CRTP. Actually the same
     * functionality could be implemented using standard inheritance
     * but we prefer the CRTP because it can be seen as the standard
     * gridtools pattern and we clearly want to avoid virtual
     * methods, etc.
     */
    template <typename DataType>
    struct host_storage : storage_interface<host_storage<DataType>> {
        typedef DataType data_t;
        typedef data_t *ptrs_t;
        typedef state_machine state_machine_t;

      private:
        data_t *m_allocated_ptr = nullptr;
        data_t *m_cpu_ptr;
        ownership m_ownership = ownership::Full;

      public:
        host_storage(host_storage const &) = delete;
        host_storage &operator=(host_storage const &) = delete;
        /*
         * @brief host_storage constructor. Just allocates enough memory on the Host.
         * @param size defines the size of the storage and the allocated space.
         */
        template <uint_t Align = 1>
        host_storage(uint_t size, uint_t offset_to_align = 0u, alignment<Align> = alignment<1u>{})
            : m_allocated_ptr(new data_t[size + Align - 1]), m_cpu_ptr(nullptr) {
            // New will align addresses according to the size(data_t)
            uint_t delta =
                ((reinterpret_cast<std::uintptr_t>(m_allocated_ptr + offset_to_align)) % (Align * sizeof(data_t))) /
                sizeof(data_t);
            m_cpu_ptr = (delta == 0) ? m_allocated_ptr : m_allocated_ptr + (Align - delta);
        }

        /*
         * @brief host_storage constructor. Does not allocate memory but uses an external pointer.
         * Reason for having this is to support externally allocated memory (e.g., from Fortran or Python).
         * @param size defines the size of the storage and the allocated space.
         * @param external_ptr a pointer to the external data
         * @param own ownership information (in this case only externalCPU is valid)
         */
        explicit host_storage(uint_t size, data_t *external_ptr, ownership own = ownership::ExternalCPU)
            : m_cpu_ptr(external_ptr),
              m_ownership(error_or_return(
                  (own == ownership::ExternalCPU), own, "ownership type must be ExternalCPU when using host_storage")) {
        }

        /*
         * @brief host_storage constructor. Allocate memory on Host and initialize the memory according to the given
         * initializer.
         * @param size defines the size of the storage and the allocated space.
         * @param initializer initialization value
         */
        template <typename Funct, uint_t Align = 1>
        host_storage(uint_t size, Funct initializer, uint_t offset_to_align = 0u, alignment<Align> a = alignment<1u>{})
            : host_storage(size, offset_to_align, a) {
            for (uint_t i = 0; i < size; ++i) {
                m_cpu_ptr[i] = initializer(i);
            }
        }

        /*
         * @brief host_storage destructor.
         */
        ~host_storage() {
            if (m_ownership == ownership::Full && m_cpu_ptr)
                delete[] m_allocated_ptr;
        }

        /*
         * @brief swap implementation for host_storage
         */
        void swap_impl(host_storage &other) {
            using std::swap;
            swap(m_cpu_ptr, other.m_cpu_ptr);
            swap(m_ownership, other.m_ownership);
        }

        /*
         * @brief retrieve the host data pointer.
         * @return data pointer
         */
        data_t *get_cpu_ptr() const {
            ASSERT_OR_THROW(m_cpu_ptr, "This storage has never been initialized");
            return m_cpu_ptr;
        }

        /*
         * @brief get_ptrs implementation for host_storage.
         */
        ptrs_t get_ptrs_impl() const { return m_cpu_ptr; }

        /*
         * @brief valid implementation for host_storage.
         */
        bool valid_impl() const { return m_cpu_ptr; }

        /*
         * @brief clone_to_device implementation for host_storage.
         */
        void clone_to_device_impl(){};

        /*
         * @brief clone_from_device implementation for host_storage.
         */
        void clone_from_device_impl(){};

        /*
         * @brief synchronization implementation for host_storage.
         */
        void sync_impl(){};

        /*
         * @brief device_needs_update implementation for host_storage.
         */
        bool device_needs_update_impl() const { return false; }

        /*
         * @brief host_needs_update implementation for host_storage.
         */
        bool host_needs_update_impl() const { return false; }

        /*
         * @brief reactivate_device_write_views implementation for host_storage.
         */
        void reactivate_device_write_views_impl() {}

        /*
         * @brief reactivate_host_write_views implementation for host_storage.
         */
        void reactivate_host_write_views_impl() {}

        /*
         * @brief get_state_machine_ptr implementation for host_storage.
         */
        state_machine *get_state_machine_ptr_impl() { return nullptr; }
    };

    // simple metafunction to check if a type is a host storage
    template <typename T>
    struct is_host_storage : std::false_type {};

    template <typename T>
    struct is_host_storage<host_storage<T>> : std::true_type {};

    /**
     * @}
     */
} // namespace gridtools

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

#include <cassert>
#include <cstddef>
#include <utility>

#include "../common/alignment.hpp"
#include "../common/state_machine.hpp"
#include "../common/storage_interface.hpp"

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
        typedef DataType *ptrs_t;
        typedef state_machine state_machine_t;

      private:
        std::unique_ptr<DataType[]> m_holder;
        DataType *m_ptr;

      public:
        /*
         * @brief host_storage constructor. Just allocates enough memory on the Host.
         * @param size defines the size of the storage and the allocated space.
         */
        template <uint_t Align = 1>
        host_storage(uint_t size, uint_t offset_to_align = 0u, alignment<Align> = alignment<1u>{})
            : m_holder(new DataType[size + Align - 1]), m_ptr(nullptr) {
            auto *allocated_ptr = m_holder.get();
            // New will align addresses according to the size(DataType)
            auto delta =
                (reinterpret_cast<std::uintptr_t>(allocated_ptr + offset_to_align) % (Align * sizeof(DataType))) /
                sizeof(DataType);
            m_ptr = delta == 0 ? allocated_ptr : allocated_ptr + (Align - delta);
        }

        /*
         * @brief host_storage constructor. Does not allocate memory but uses an external pointer.
         * Reason for having this is to support externally allocated memory (e.g., from Fortran or Python).
         * @param size defines the size of the storage and the allocated space.
         * @param external_ptr a pointer to the external data
         * @param own ownership information (in this case only externalCPU is valid)
         */
        host_storage(uint_t size, DataType *external_ptr, ownership own = ownership::ExternalCPU)
            : m_ptr(external_ptr) {
            assert(external_ptr);
            assert(own == ownership::ExternalCPU);
        }

        /*
         * @brief host_storage constructor. Allocate memory on Host and initialize the memory according to the given
         * initializer.
         * @param size defines the size of the storage and the allocated space.
         * @param initializer initialization value
         */
        template <typename Fun, uint_t Align = 1>
        host_storage(uint_t size, Fun &&initializer, uint_t offset_to_align = 0u, alignment<Align> a = alignment<1u>{})
            : host_storage(size, offset_to_align, a) {
            for (uint_t i = 0; i < size; ++i)
                m_ptr[i] = initializer(i);
        }

        /*
         * @brief swap implementation for host_storage
         */
        void swap_impl(host_storage &other) {
            using std::swap;
            swap(m_holder, other.m_holder);
            swap(m_ptr, other.m_ptr);
        }

        /*
         * @brief retrieve the host data pointer.
         * @return data pointer
         */
        DataType *get_cpu_ptr() const { return m_ptr; }

        /*
         * @brief get_ptrs implementation for host_storage.
         */
        DataType *get_ptrs_impl() const { return m_ptr; }

        /*
         * @brief valid implementation for host_storage.
         */
        bool valid_impl() const { return true; }

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

/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <atomic>
#include <utility>

#include "../../common/gt_assert.hpp"
#include "../common/state_machine.hpp"
#include "../common/storage_interface.hpp"

namespace gridtools {

    /*
     * @brief The Mic storage implementation. This class owns the pointer
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
    struct mc_storage : storage_interface<mc_storage<DataType>> {
        typedef DataType data_t;
        typedef DataType *ptrs_t;
        typedef state_machine state_machine_t;

      private:
        std::unique_ptr<DataType, std::integral_constant<decltype(&free), &free>> m_holder;
        DataType *m_ptr;

      public:
        /*
         * @brief mc_storage constructor. Allocates data aligned to 2MB pages (to encourage the system to use
         * transparent huge pages) and adds an additional samll offset which changes for every allocation to reduce the
         * risk of L1 cache set conflicts.
         * @param size defines the size of the storage and the allocated space.
         */
        template <uint_t Align = 1>
        mc_storage(uint_t size, uint_t offset_to_align = 0u, alignment<Align> = alignment<1u>{}) {
            // New will align addresses according to the size(data_t)
            static std::atomic<uint_t> s_data_offset(64);
            uint_t data_offset = s_data_offset.load(std::memory_order_relaxed);
            uint_t data_type_offset = 0;
            uint_t next_data_offset;
            do {
                data_type_offset = data_offset / sizeof(DataType);
                next_data_offset = 2 * data_offset;
                if (next_data_offset > 8192)
                    next_data_offset = 64;
            } while (!s_data_offset.compare_exchange_weak(data_offset, next_data_offset, std::memory_order_relaxed));

            DataType *allocated_ptr;
            if (posix_memalign(reinterpret_cast<void **>(&allocated_ptr),
                    2 * 1024 * 1024,
                    (size + (data_type_offset + Align)) * sizeof(data_t)))
                throw std::bad_alloc();

            uint_t delta =
                ((reinterpret_cast<std::uintptr_t>(allocated_ptr + offset_to_align)) % (Align * sizeof(data_t))) /
                sizeof(data_t);
            m_holder.reset(allocated_ptr);
            m_ptr =
                (delta == 0) ? allocated_ptr + data_type_offset : allocated_ptr + (data_type_offset + Align - delta);
        }

        /*
         * @brief mc_storage constructor. Does not allocate memory but uses an external pointer.
         * Reason for having this is to support externally allocated memory (e.g., from Fortran or Python).
         * @param size defines the size of the storage and the allocated space.
         * @param external_ptr a pointer to the external data
         * @param own ownership information (in this case only externalCPU is valid)
         */
        mc_storage(uint_t size, data_t *external_ptr, ownership own = ownership::external_cpu) : m_ptr(external_ptr) {
            assert(own == ownership::external_cpu);
        }

        /*
         * @brief mc_storage constructor. Allocate memory on Mic and initialize the memory according to the given
         * initializer.
         * @param size defines the size of the storage and the allocated space.
         * @param initializer initialization value
         */
        template <typename Fun, uint_t Align = 1>
        mc_storage(uint_t size, Fun &&initializer, uint_t offset_to_align = 0u, alignment<Align> a = alignment<1u>{})
            : mc_storage(size, offset_to_align, a) {
            for (uint_t i = 0; i < size; ++i)
                m_ptr[i] = initializer(i);
        }

        /*
         * @brief swap implementation for mc_storage
         */
        void swap_impl(mc_storage &other) {
            using std::swap;
            swap(m_holder, other.m_holder);
            swap(m_ptr, other.m_ptr);
        }

        /*
         * @brief retrieve the mc data pointer.
         * @return data pointer
         */
        DataType *get_cpu_ptr() const { return m_ptr; }

        /*
         * @brief get_ptrs implementation for mc_storage.
         */
        DataType *get_ptrs_impl() const { return m_ptr; }

        /*
         * @brief valid implementation for mc_storage.
         */
        bool valid_impl() const { return true; }

        /*
         * @brief clone_to_device implementation for mc_storage.
         */
        void clone_to_device_impl(){};

        /*
         * @brief clone_from_device implementation for mc_storage.
         */
        void clone_from_device_impl(){};

        /*
         * @brief synchronization implementation for mc_storage.
         */
        void sync_impl(){};

        /*
         * @brief device_needs_update implementation for mc_storage.
         */
        bool device_needs_update_impl() const { return false; }

        /*
         * @brief host_needs_update implementation for mc_storage.
         */
        bool host_needs_update_impl() const { return false; }

        /*
         * @brief reactivate_device_write_views implementation for mc_storage.
         */
        void reactivate_device_write_views_impl() {}

        /*
         * @brief reactivate_host_write_views implementation for mc_storage.
         */
        void reactivate_host_write_views_impl() {}

        /*
         * @brief get_state_machine_ptr implementation for mc_storage.
         */
        state_machine *get_state_machine_ptr_impl() { return nullptr; }
    };

    // simple metafunction to check if a type is a mc storage
    template <typename T>
    struct is_mc_storage : std::false_type {};

    template <typename T>
    struct is_mc_storage<mc_storage<T>> : std::true_type {};
} // namespace gridtools

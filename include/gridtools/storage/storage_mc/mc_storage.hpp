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

#include <atomic>
#include <utility>

#include "../../common/gt_assert.hpp"
#include "../../common/hugepage_alloc.hpp"
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
        typedef state_machine state_machine_t;

      private:
        std::unique_ptr<void, std::integral_constant<decltype(&hugepage_free), &hugepage_free>> m_holder;
        DataType *m_ptr;

      public:
        /*
         * @brief mc_storage constructor. Allocates data aligned to 2MB pages (to encourage the system to use
         * transparent huge pages) and adds an additional samll offset which changes for every allocation to reduce the
         * risk of L1 cache set conflicts.
         * @param size defines the size of the storage and the allocated space.
         */
        template <uint_t Align = 1>
        mc_storage(uint_t size, uint_t offset_to_align = 0u, alignment<Align> = alignment<1u>{})
            : m_holder(hugepage_alloc((size + Align) * sizeof(DataType))) {
            constexpr auto byte_alignment = Align * sizeof(DataType);
            auto byte_offset = offset_to_align * sizeof(DataType);
            auto address_to_align = reinterpret_cast<std::uintptr_t>(m_holder.get()) + byte_offset;
            m_ptr = reinterpret_cast<DataType *>(
                (address_to_align + byte_alignment - 1) / byte_alignment * byte_alignment - byte_offset);
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

        DataType *get_target_ptr() const { return m_ptr; }
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
         * @brief reactivate_target_write_views implementation for mc_storage.
         */
        void reactivate_target_write_views_impl() {}

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

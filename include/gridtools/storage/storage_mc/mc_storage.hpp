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
    namespace mc_storage_impl_ {
        template <typename DataType>
        class mc_storage : public storage_interface<mc_storage<DataType>> {
          private:
            std::unique_ptr<void, std::integral_constant<decltype(&hugepage_free), &hugepage_free>> m_holder;
            DataType *m_ptr;

          public:
            using data_t = DataType;

            /*
             * @brief mc_storage constructor. Allocates data aligned to 2MB pages (to encourage the system to use
             * transparent huge pages) and adds an additional samll offset which changes for every allocation to reduce
             * the risk of L1 cache set conflicts.
             * @param size defines the size of the storage and the allocated space.
             */
            template <uint_t Align>
            mc_storage(uint_t size, uint_t offset_to_align, alignment<Align>)
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
            mc_storage(uint_t size, data_t *external_ptr, ownership own = ownership::external_cpu)
                : m_ptr(external_ptr) {
                assert(external_ptr);
                assert(own == ownership::external_cpu);
            }

            /*
             * @brief retrieve the mc data pointer.
             * @return data pointer
             */
            DataType *get_cpu_ptr_impl() const { return m_ptr; }
        };
    } // namespace mc_storage_impl_

    using mc_storage_impl_::mc_storage;
} // namespace gridtools

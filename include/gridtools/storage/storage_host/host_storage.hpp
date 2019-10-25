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

#include <cassert>
#include <cstddef>
#include <utility>

#include "../common/alignment.hpp"
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
    namespace host_storage_impl_ {
        template <typename DataType>
        class host_storage : public storage_interface<host_storage<DataType>> {
            std::unique_ptr<DataType[]> m_holder;
            DataType *m_ptr;

          public:
            using data_t = DataType;

            /*
             * @brief host_storage constructor. Just allocates enough memory on the Host.
             * @param size defines the size of the storage and the allocated space.
             */
            template <uint_t Align>
            host_storage(uint_t size, uint_t offset_to_align, alignment<Align>)
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
            host_storage(uint_t, DataType *external_ptr, ownership own = ownership::external_cpu)
                : m_ptr(external_ptr) {
                assert(external_ptr);
                assert(own == ownership::external_cpu);
            }

            /*
             * @brief retrieve the host data pointer.
             * @return data pointer
             */
            DataType *get_cpu_ptr_impl() const { return m_ptr; }
        };
    } // namespace host_storage_impl_

    using host_storage_impl_::host_storage;
    /**
     * @}
     */
} // namespace gridtools

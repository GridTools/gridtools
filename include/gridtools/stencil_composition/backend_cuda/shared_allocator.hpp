/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "../../common/defs.hpp"
#include "../../common/host_device.hpp"

namespace gridtools {

    class shared_allocator {
      private:
        uint_t m_offset = 0; // in bytes

      public:
        template <typename T>
        struct lazy_alloc {
            using element_type = T;

            uint_t m_offset;
            GT_FUNCTION_DEVICE T *operator()() const {
                extern __shared__ char ij_cache_shm[];
                return reinterpret_cast<T *>(ij_cache_shm + m_offset);
            }

            friend GT_FORCE_INLINE lazy_alloc operator+(lazy_alloc l, int_t r) {
                l.m_offset += r * sizeof(T);
                return l;
            }
        };

        /**
         * \param sz size of allocation in number of elements
         */
        template <class T, uint_t Alignment = sizeof(T)>
        lazy_alloc<T> allocate(uint_t sz) {
            auto aligned = (m_offset + Alignment - 1) / Alignment * Alignment;
            m_offset = aligned + sz * sizeof(T);
            return {aligned};
        }

        uint_t size() const { return m_offset; }
    };

} // namespace gridtools

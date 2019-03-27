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
        uint_t m_offset = 0;

      public:
        /**
         * \tparam Alignment required alignment in bytes
         * \param sz size of allocation in bytes
         */
        template <uint_t Alignment>
        int_t allocate(uint_t sz) {
            auto aligned = (m_offset + Alignment - 1) / Alignment * Alignment;
            m_offset = aligned + sz;
            return aligned;
        }

        uint_t size() const { return m_offset; }
    };

} // namespace gridtools

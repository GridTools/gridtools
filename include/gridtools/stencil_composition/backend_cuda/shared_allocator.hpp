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

    class SharedAllocator {
      private:
        uint_t m_offset = 0;

      public:
        template <class T>
        int_t allocate(size_t sz) {
            auto aligned = (m_offset + sizeof(T) - 1) / sizeof(T) * sizeof(T);
            m_offset = aligned + sz * sizeof(T);
            return aligned / sizeof(T);
        }

        uint_t size() const { return m_offset; }
    };

} // namespace gridtools

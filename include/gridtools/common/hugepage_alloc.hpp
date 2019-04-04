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
#include <cstdlib>
#include <new>

namespace gridtools {

    /**
     * @brief Allocates huge page memory (if GT_NO_HUGETLB is not defined) and shifts allocations by some bytes to
     * reduce cache set conflicts.
     */
    inline void *hugepage_alloc(std::size_t size) {
        static std::atomic<std::size_t> s_offset(64);
        auto offset = s_offset.load(std::memory_order_relaxed);
        auto next_offset = offset;
        while (!s_offset.compare_exchange_weak(
            next_offset, 2 * next_offset <= 4096 ? 2 * next_offset : 64, std::memory_order_relaxed)) {
        }

        void *ptr;
        if (posix_memalign(&ptr, 2 * 1024 * 1024, size + offset))
            throw std::bad_alloc();

        ptr = static_cast<char *>(ptr) + offset;
        static_cast<std::size_t *>(ptr)[-1] = offset;
        return ptr;
    }

    /**
     * @brief Frees memory allocated by hugepage_alloc.
     */
    inline void hugepage_free(void *ptr) {
        if (!ptr)
            return;
        std::size_t offset = static_cast<std::size_t *>(ptr)[-1];
        ptr = static_cast<char *>(ptr) - offset;
        free(ptr);
    }

} // namespace gridtools

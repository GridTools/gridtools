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
#include <new>

#include <sys/mman.h>

namespace gridtools {

    /**
     * @brief Allocates huge page memory (if GT_NO_HUGETLB is not defined) and shifts allocations by some bytes to
     * reduce cache set conflicts.
     */
    void *hugepage_alloc(std::size_t size) {
        static std::atomic<std::size_t> s_offset(64);
        auto offset = s_offset.load(std::memory_order_relaxed);
        auto next_offset = offset;
        while (!s_offset.compare_exchange_weak(
            next_offset, 2 * next_offset <= 4096 ? 2 * next_offset : 64, std::memory_order_relaxed))
            ;

        static constexpr auto prot = PROT_READ | PROT_WRITE;
#ifdef GT_NO_HUGETLB
        static constexpr auto flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
#else
        static constexpr auto flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_HUGETLB;
#endif

        std::size_t alloc_size = size + offset;
        void *ptr = mmap(nullptr, alloc_size, prot, flags, -1, 0);
        if (ptr == MAP_FAILED)
            throw std::bad_alloc();

        ptr = static_cast<void *>(static_cast<char *>(ptr) + offset);
        static_cast<std::size_t *>(ptr)[-1] = offset;
        static_cast<std::size_t *>(ptr)[-2] = alloc_size;
        return ptr;
    }

    /**
     * @brief Frees memory allocated by hugepage_alloc.
     */
    void hugepage_free(void *ptr) {
        if (!ptr)
            return;
        std::size_t offset = static_cast<std::size_t *>(ptr)[-1];
        std::size_t alloc_size = static_cast<std::size_t *>(ptr)[-2];
        ptr = static_cast<void *>(static_cast<char *>(ptr) - offset);
        munmap(ptr, alloc_size);
    }

} // namespace gridtools

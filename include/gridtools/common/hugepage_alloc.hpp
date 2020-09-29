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
#include <fstream>
#include <new>
#if __cpp_lib_int_pow2 >= 202002L
#include <bit>
#endif

namespace gridtools {
    namespace hugepage_alloc_impl_ {
        inline std::size_t ilog2(std::size_t i) {
#if __cpp_lib_int_pow2 >= 202002L
            return std::bit_width(t) - 1;
#else
            std::size_t log = 0;
            while (i >>= 1)
                ++log;
            return log;
#endif
        }

        inline std::size_t cache_line_size() {
            std::size_t value = 64; // default value for x86-64 archs
#if __linux__
            std::ifstream f("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size");
            if (f.is_open())
                f >> value;
#endif
            return value;
        }

        inline std::size_t cache_sets() {
            std::size_t value = 64; // default value for (most?) x86-64 archs
#if __linux__
            std::ifstream f("/sys/devices/system/cpu/cpu0/cache/index0/number_of_sets");
            if (f.is_open())
                f >> value;
#endif
            return value;
        }
    } // namespace hugepage_alloc_impl_

    /**
     * @brief Allocates huge page memory (if GT_NO_HUGETLB is not defined) and shifts allocations by some bytes to
     * reduce cache set conflicts.
     */
    inline void *hugepage_alloc(std::size_t size) {
        static std::atomic<std::size_t> s_offset(0);
        static const std::size_t cache_line_size = hugepage_alloc_impl_::cache_line_size();
        static const std::size_t cache_set_size = hugepage_alloc_impl_::cache_sets();

        std::size_t offset =
            ((s_offset++ % cache_set_size) << hugepage_alloc_impl_::ilog2(cache_line_size)) + cache_line_size;

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

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
#ifdef __linux__
#include <regex>
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
            std::ifstream file("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size");
            if (file.is_open())
                file >> value;
#endif
            return value;
        }

        inline std::size_t cache_sets() {
            std::size_t value = 64; // default value for (most?) x86-64 archs
#if __linux__
            std::ifstream file("/sys/devices/system/cpu/cpu0/cache/index0/number_of_sets");
            if (file.is_open())
                file >> value;
#endif
            return value;
        }

        inline std::size_t hugepage_size() {
            std::size_t value = 2 * 1024 * 1024; // 2MB default on most systems
#if __linux__
            std::ifstream file("/proc/meminfo");
            if (file.is_open()) {
                std::regex re("^Hugepagesize: *([0-9]+) *kB");
                std::string line;
                while (std::getline(file, line)) {
                    std::smatch match;
                    if (std::regex_match(line, match, re)) {
                        value = std::stoll(match[1].str()) * 1024;
                        break;
                    }
                }
            }
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
        static const std::size_t hugepage_size = hugepage_alloc_impl_::hugepage_size();

        std::size_t offset =
            ((s_offset++ % cache_set_size) << hugepage_alloc_impl_::ilog2(cache_line_size)) + cache_line_size;

        void *ptr;
        if (posix_memalign(&ptr, hugepage_size, size + offset))
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

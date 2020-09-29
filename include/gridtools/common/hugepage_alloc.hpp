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
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <new>
#include <stdexcept>
#if __cpp_lib_int_pow2 >= 202002L
#include <bit>
#endif
#ifdef __linux__
#include <iostream>
#include <regex>

#include <sys/mman.h>
#include <unistd.h>
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
#ifdef __linux__
            std::ifstream file("/sys/devices/system/cpu/cpu0/cache/index0/number_of_sets");
            if (file.is_open())
                file >> value;
#endif
            return value;
        }

        enum class hugepage_mode { disabled, transparent, explicit_allocation };

        inline hugepage_mode hugepage_mode_from_env() {
            const char *env_value = std::getenv("GT_HUGEPAGE_MODE");
            if (!env_value || std::strcmp(env_value, "transparent") == 0)
                return hugepage_mode::transparent;
            if (std::strcmp(env_value, "disabled") == 0)
                return hugepage_mode::disabled;
            if (std::strcmp(env_value, "explicit") == 0)
                return hugepage_mode::explicit_allocation;
            return hugepage_mode::transparent;
        }

        inline std::size_t hugepage_size() {
#ifdef __linux__
            std::ifstream file("/proc/meminfo");
            if (file.is_open()) {
                std::regex re("^Hugepagesize: *([0-9]+) *kB$");
                std::string line;
                while (std::getline(file, line)) {
                    std::smatch match;
                    if (std::regex_match(line, match, re))
                        return std::stoll(match[1].str()) * 1024;
                }
            }
#endif
            return 2 * 1024 * 1024; // 2MB is the default on most systems
        }

        struct ptr_metadata {
            std::size_t offset, full_size;
            hugepage_mode mode;
        };

    } // namespace hugepage_alloc_impl_

    /**
     * @brief Allocates huge page memory (if GT_NO_HUGETLB is not defined) and shifts allocations by some bytes to
     * reduce cache set conflicts.
     */
    inline void *hugepage_alloc(std::size_t size) {
        static std::atomic<std::size_t> s_offset(0);
        static const std::size_t cache_line_size = hugepage_alloc_impl_::cache_line_size();
        static const std::size_t cache_sets = hugepage_alloc_impl_::cache_sets();
        static const std::size_t hugepage_size = hugepage_alloc_impl_::hugepage_size();
#ifdef __linux__
        static const std::size_t page_size = sysconf(_SC_PAGESIZE);
#else
        static const std::size_t page_size = 4096; // just assume 4KB, default on most systems
#endif
        assert(cache_line_size >= sizeof(hugepage_alloc_impl_::ptr_metadata));

        std::size_t offset =
            ((s_offset++ % cache_sets) << hugepage_alloc_impl_::ilog2(cache_line_size)) + cache_line_size;

        std::size_t full_size = size + offset;

        void *ptr = nullptr;
        auto mode = hugepage_alloc_impl_::hugepage_mode_from_env();
        switch (mode) {
        case hugepage_alloc_impl_::hugepage_mode::disabled:
            // here we just align to small/normal page size
            full_size = ((full_size + page_size - 1) / page_size) * page_size;
            if (posix_memalign(&ptr, page_size, full_size))
                throw std::bad_alloc();
            break;
#ifdef __linux__
        case hugepage_alloc_impl_::hugepage_mode::transparent:
            // here we try to get transparent huge pages
            full_size = ((full_size + hugepage_size - 1) / hugepage_size) * hugepage_size;
            ptr = mmap(nullptr, full_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            if (ptr == MAP_FAILED)
                throw std::bad_alloc();
            madvise(ptr, full_size, MADV_HUGEPAGE);
            break;
        case hugepage_alloc_impl_::hugepage_mode::explicit_allocation:
            // here we force huge page allocation (fails with a bus error if none are available)
            full_size = ((full_size + hugepage_size - 1) / hugepage_size) * hugepage_size;
            ptr = mmap(nullptr,
                full_size,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_NORESERVE,
                -1,
                0);
            if (ptr == MAP_FAILED)
                throw std::bad_alloc();
            break;
#else
        case hugepage_alloc_impl_::hugepage_mode::transparent:
        case hugepage_alloc_impl_::hugepage_mode::explicit_allocation:
            // here we hope that aligning to hugepage_size will return transparent huge pages
            full_size = ((full_size + hugepage_size - 1) / hugepage_size) * hugepage_size;
            if (posix_memalign(&ptr, hugepage_size, full_size))
                throw std::bad_alloc();
            break;
#endif
        }

        ptr = static_cast<char *>(ptr) + offset;
        static_cast<hugepage_alloc_impl_::ptr_metadata *>(ptr)[-1] = {offset, full_size, mode};
        return ptr;
    }

    /**
     * @brief Frees memory allocated by hugepage_alloc.
     */
    inline void hugepage_free(void *ptr) {
        if (!ptr)
            return;
        auto &metadata = static_cast<hugepage_alloc_impl_::ptr_metadata *>(ptr)[-1];
        ptr = static_cast<char *>(ptr) - metadata.offset;
        switch (metadata.mode) {
#ifdef __linux__
        case hugepage_alloc_impl_::hugepage_mode::transparent:
        case hugepage_alloc_impl_::hugepage_mode::explicit_allocation:
            if (munmap(ptr, metadata.full_size))
                throw std::bad_alloc();
            break;
#endif
        default:
            free(ptr);
            break;
        }
    }

} // namespace gridtools

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

#ifdef __linux__
        inline std::size_t parse_proc_meminfo(std::string const &regex) {
            std::ifstream file("/proc/meminfo");
            if (file.is_open()) {
                std::regex re(regex);
                std::string line;
                while (std::getline(file, line)) {
                    std::smatch match;
                    if (std::regex_match(line, match, re))
                        return std::stoll(match[1].str());
                }
            }
            throw std::runtime_error("regex '" + regex + "' did not match /proc/meminfo");
        }
#endif

        inline std::size_t hugepage_size() {
#ifdef GT_NO_HUGETLB
            return 1;
#else
#ifdef __linux__
            return parse_proc_meminfo("^Hugepagesize: *([0-9]+) *kB$") * 1024;
#else
            return 2 * 1024 * 1024; // 2MB is the default on most systems
#endif
#endif
        }

#ifdef __linux__
        inline std::size_t free_hugepages() { return parse_proc_meminfo("^HugePages_Free: *([0-9]+) *$"); }
#endif
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
        assert(cache_line_size >= 2 * sizeof(std::size_t));

        std::size_t offset =
            ((s_offset++ % cache_sets) << hugepage_alloc_impl_::ilog2(cache_line_size)) + cache_line_size;
        std::size_t full_size = ((size + offset + hugepage_size - 1) / hugepage_size) * hugepage_size;

        void *ptr = nullptr;
#if defined(__linux__) && !defined(GT_NO_HUGETLB)
        // we test if explicit huge pages are available to make sure that mmap does not trigger a bus error there is
        // probably no full-system thread-safe way to do so, so we might still run into one, but the main functionality
        // is to avoid bus errors if explicit huge pages are not enabled on the system and thus free_hugepages() returns
        // zero always
        if (hugepage_alloc_impl_::free_hugepages() * hugepage_size >= full_size) {
            ptr = mmap(nullptr,
                full_size,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_NORESERVE,
                -1,
                0);
            if (ptr == MAP_FAILED)
                throw std::bad_alloc();
        } else {
            // here we try to get transparent huge pages
            ptr = mmap(nullptr, full_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            if (ptr == MAP_FAILED)
                throw std::bad_alloc();
            madvise(ptr, full_size, MADV_HUGEPAGE);
        }
#elif !defined(GT_NO_HUGETLB)
        // here we hope that aligning to hugepage_size will return transparent huge pages
        if (posix_memalign(&ptr, hugepage_size, full_size))
            throw std::bad_alloc();
#else
        // here we just align to cache line size
        if (posix_memalign(&ptr, cache_line_size, full_size))
            throw std::bad_alloc();
#endif

        ptr = static_cast<char *>(ptr) + offset;
        static_cast<std::size_t *>(ptr)[-1] = offset;
        static_cast<std::size_t *>(ptr)[-2] = full_size;
        return ptr;
    }

    /**
     * @brief Frees memory allocated by hugepage_alloc.
     */
    inline void hugepage_free(void *ptr) {
        if (!ptr)
            return;
        std::size_t offset = static_cast<std::size_t *>(ptr)[-1];
        std::size_t full_size = static_cast<std::size_t *>(ptr)[-2];
        ptr = static_cast<char *>(ptr) - offset;
#if defined(__linux__) && !defined(GT_NO_HUGETLB)
        if (munmap(ptr, full_size))
            throw std::bad_alloc();
#else
        free(ptr);
#endif
    }

} // namespace gridtools

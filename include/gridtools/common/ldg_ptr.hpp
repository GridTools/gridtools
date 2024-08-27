/*
 * GridTools
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include "defs.hpp"
#include "host_device.hpp"

#ifdef GT_CUDACC
#include "cuda_type_traits.hpp"
#endif

namespace gridtools {

#ifdef GT_CUDACC
    namespace impl_ {

        template <class T>
        struct ldg_ptr {
            T const *m_ptr;

            static_assert(is_texture_type<T>::value);

            GT_FUNCTION constexpr T operator*() const {
#ifdef GT_CUDA_ARCH
                return __ldg(m_ptr);
#else
                return *m_ptr;
#endif
            }

            GT_FUNCTION constexpr ldg_ptr &operator+=(std::ptrdiff_t diff) {
                m_ptr += diff;
                return *this;
            }

            GT_FUNCTION constexpr ldg_ptr &operator-=(std::ptrdiff_t diff) {
                m_ptr -= diff;
                return *this;
            }

            friend GT_FUNCTION constexpr bool operator==(ldg_ptr const &a, ldg_ptr const &b) {
                return a.m_ptr == b.m_ptr;
            }
            friend GT_FUNCTION constexpr bool operator==(ldg_ptr const &a, T const *b) { return a.m_ptr == b; }
            friend GT_FUNCTION constexpr bool operator==(T const *a, ldg_ptr const &b) { return a == b.m_ptr; }

            friend GT_FUNCTION constexpr bool operator!=(ldg_ptr const &a, ldg_ptr const &b) {
                return a.m_ptr != b.m_ptr;
            }
            friend GT_FUNCTION constexpr bool operator!=(ldg_ptr const &a, T const *b) { return a.m_ptr != b; }
            friend GT_FUNCTION constexpr bool operator!=(T const *a, ldg_ptr const &b) { return a != b.m_ptr; }

            friend GT_FUNCTION constexpr ldg_ptr &operator++(ldg_ptr &ptr) {
                ++ptr.m_ptr;
                return ptr;
            }

            friend GT_FUNCTION constexpr ldg_ptr &operator--(ldg_ptr &ptr) {
                --ptr.m_ptr;
                return ptr;
            }

            friend GT_FUNCTION constexpr ldg_ptr operator++(ldg_ptr &ptr, int) {
                ldg_ptr p = ptr;
                ++ptr.m_ptr;
                return p;
            }

            friend GT_FUNCTION constexpr ldg_ptr operator--(ldg_ptr &ptr, int) {
                ldg_ptr p = ptr;
                --ptr.m_ptr;
                return p;
            }

            friend GT_FUNCTION constexpr ldg_ptr operator+(ldg_ptr const &ptr, std::ptrdiff_t diff) {
                return {ptr.m_ptr + diff};
            }

            friend GT_FUNCTION constexpr ldg_ptr operator-(ldg_ptr const &ptr, std::ptrdiff_t diff) {
                return {ptr.m_ptr - diff};
            }

            friend GT_FUNCTION constexpr std::ptrdiff_t operator-(ldg_ptr const &ptr, ldg_ptr const &other) {
                return ptr.m_ptr - other.m_ptr;
            }
        };
    } // namespace impl_

    template <class T>
    GT_FUNCTION constexpr std::enable_if_t<is_texture_type<T>::value, impl_::ldg_ptr<T>> as_ldg_ptr(T const *ptr) {
        return {ptr};
    }

#endif

    template <class T>
    GT_FUNCTION constexpr T &&as_ldg_ptr(T &&value) {
        return std::forward<T>(value);
    }

} // namespace gridtools

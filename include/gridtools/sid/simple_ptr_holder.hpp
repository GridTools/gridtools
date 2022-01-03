/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <utility>

#include "../common/host_device.hpp"

namespace gridtools {
    namespace sid {
        template <class T>
        struct simple_ptr_holder {
            T m_val;
            simple_ptr_holder() = default;

            GT_FORCE_INLINE constexpr simple_ptr_holder(T const& val) : m_val(val) {}

            GT_FORCE_INLINE constexpr T const &operator()() const { return m_val; }
        };

        template <class T, class Arg>
        constexpr auto operator+(simple_ptr_holder<T> const &obj, Arg &&arg) {
            return simple_ptr_holder(obj.m_val + std::forward<Arg>(arg));
        }

        template <class T, class Arg>
        constexpr auto operator+(simple_ptr_holder<T> &&obj, Arg &&arg) {
            return simple_ptr_holder(std::move(obj.m_val) + std::forward<Arg>(arg));
        }
    } // namespace sid
} // namespace gridtools

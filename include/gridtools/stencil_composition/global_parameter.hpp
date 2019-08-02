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

#include <type_traits>
#include <utility>

#include "../common/defs.hpp"
#include "../common/host_device.hpp"

namespace gridtools {
    namespace global_parameter_impl_ {
        struct ptr_diff {};

        template <class T>
        struct global_parameter {
            GT_STATIC_ASSERT(std::is_trivially_copyable<T>(), "global parameter should be trivially copyable");

            T m_value;

            GT_CONSTEXPR GT_FUNCTION global_parameter operator()() const { return *this; }
            GT_CONSTEXPR GT_FUNCTION T const &operator*() const { return m_value; }

            friend GT_FUNCTION global_parameter operator+(global_parameter obj, ptr_diff) { return obj; }
            friend global_parameter sid_get_origin(global_parameter const &obj) { return obj; }
            friend ptr_diff sid_get_ptr_diff(global_parameter) { return {}; }
        };
    } // namespace global_parameter_impl_

    template <class U, class T = U>
    using global_parameter = global_parameter_impl_::global_parameter<T>;

    template <class = void, class T>
    global_parameter<T> make_global_parameter(T val) {
        return {std::move(val)};
    }

    template <class T>
    void update_global_parameter(global_parameter<T> &dst, T src) {
        dst = {std::move(src)};
    }
} // namespace gridtools

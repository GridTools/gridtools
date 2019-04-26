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

#include <tuple>
#include <type_traits>
#include <utility>

#include "../host_device.hpp"

namespace gridtools {
    /**
     *  `std::forward`/`std::move` versions that are guaranteed to be not constexpr
     */
    namespace const_expr {
        template <class T>
        GT_CONSTEXPR GT_HOST_DEVICE typename std::remove_reference<T>::type &&move(T &&obj) noexcept {
            return static_cast<typename std::remove_reference<T>::type &&>(obj);
        }
        template <class T>
        GT_CONSTEXPR GT_HOST_DEVICE T &&forward(typename std::remove_reference<T>::type &obj) noexcept {
            return static_cast<T &&>(obj);
        }
        template <class T>
        GT_CONSTEXPR GT_HOST_DEVICE T &&forward(typename std::remove_reference<T>::type &&obj) noexcept {
            static_assert(
                !std::is_lvalue_reference<T>::value, "Error: obj is instantiated with an lvalue reference type");
            return static_cast<T &&>(obj);
        }

    } // namespace const_expr
} // namespace gridtools

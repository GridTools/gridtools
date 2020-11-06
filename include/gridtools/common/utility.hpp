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

#include "defs.hpp"
#include "host_device.hpp"

namespace gridtools {
    /**
     *  `std::forward`/`std::move` versions that are guaranteed to be not constexpr. They are needed because
     *  some compilers, especially nvcc have problems with functions that return references in constexpr functions,
     *  if they are not used in constexpr context. As the `std` versions are constexpr, we must have separate
     *  functions that are constexpr only if the compiler is known to not mess up with them.
     */
    namespace wstd {
        template <class T>
        GT_CONSTEXPR GT_HOST_DEVICE std::remove_reference_t<T> &&move(T &&obj) noexcept {
            return static_cast<std::remove_reference_t<T> &&>(obj);
        }
        template <class T>
        GT_CONSTEXPR GT_HOST_DEVICE T &&forward(std::remove_reference_t<T> &obj) noexcept {
            return static_cast<T &&>(obj);
        }
        template <class T>
        GT_CONSTEXPR GT_HOST_DEVICE T &&forward(std::remove_reference_t<T> &&obj) noexcept {
            static_assert(
                !std::is_lvalue_reference<T>::value, "Error: obj is instantiated with an lvalue reference type");
            return static_cast<T &&>(obj);
        }

    } // namespace wstd
} // namespace gridtools

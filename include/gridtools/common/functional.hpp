/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
/**
@file
@brief Unstructured collection of small generic purpose functors and related helpers.
*/
#pragma once

#include <utility>

#include "host_device.hpp"

namespace gridtools {
    /** \ingroup common
        @{
        \defgroup functional Functional
        @{
    */
    /// Forward the args to constructor.
    //
    template <class T>
    struct ctor {
        template <class... Args>
        GT_FORCE_INLINE constexpr T operator()(Args &&...args) const {
            return {std::forward<Args>(args)...};
        }
    };

    /// Do nothing.
    //
    struct noop {
        template <class... Args>
        GT_FORCE_INLINE constexpr void operator()(Args &&...) const {}
    };

    /// Perfectly forward the argument.
    //
    struct identity {
        template <class Arg>
        GT_FORCE_INLINE constexpr Arg operator()(Arg &&arg) const {
            return arg;
        }
    };

    /// Copy the argument.
    //
    struct clone {
        template <class Arg>
        GT_FORCE_INLINE constexpr Arg operator()(Arg const &arg) const {
            return arg;
        }
    };

    template <class... Fs>
    struct overload : Fs... {
        constexpr overload(Fs... fs) : Fs(std::move(fs))... {}
        using Fs::operator()...;
    };
    /** @} */
    /** @} */
} // namespace gridtools

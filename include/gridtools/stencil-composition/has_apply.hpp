/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <tuple>
#include <type_traits>

#include "../meta/type_traits.hpp"

namespace gridtools {
    namespace _impl {
        template <class T, class ExtraArgs, class = void>
        struct has_apply_impl : std::false_type {};

        template <class T, class... ExtraArgs>
        struct has_apply_impl<T,
            std::tuple<ExtraArgs...>,
            void_t<decltype(T::apply(std::declval<int &>(), std::declval<ExtraArgs>()...))>> : std::true_type {};
    } // namespace _impl

    /**
     * @struct has_apply
     * Meta function testing if a functor has a specific apply method
     * (note that the meta function does consider overload resolution as well)
     */
    template <class T, class... ExtraArgs>
    struct has_apply : _impl::has_apply_impl<T, std::tuple<ExtraArgs...>> {};
} // namespace gridtools

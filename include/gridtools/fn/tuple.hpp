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

#include <tuple>
#include <type_traits>
#include <utility>

#include "../meta.hpp"

namespace gridtools::fn {
    namespace tuple_impl_ {
        struct undefined {};
        undefined fn_make_tuple(...) { return {}; }
        undefined fn_tuple_get(...) { return {}; }

        constexpr auto make_tuple = []<class... Args>(Args && ... args) {
            if constexpr (std::is_same_v<decltype(fn_make_tuple(std::forward<Args>(args)...)), undefined>)
                return std::tuple(std::forward<Args>(args)...);
            else
                return fn_make_tuple(std::forward<Args>(args)...);
        };

        template <size_t I>
        constexpr auto tuple_get = []<class Arg>(Arg &&arg) {
            if constexpr (std::is_same_v<decltype(fn_tuple_get(meta::constant<I>, std::forward<Arg>(arg))), undefined>)
                return std::get<I>(std::forward<Arg>(arg));
            else
                return fn_tuple_get(meta::constant<I>, std::forward<Arg>(arg));
        };
    } // namespace tuple_impl_
    using tuple_impl_::make_tuple;
    using tuple_impl_::tuple_get;
} // namespace gridtools::fn

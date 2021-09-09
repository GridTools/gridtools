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

namespace gridtools::fn {
    namespace arith_impl_ {
        struct undefined {};

        undefined fn_plus(...) { return {}; }
        undefined fn_minus(...) { return {}; }
        undefined fn_multiplies(...) { return {}; }
        undefined fn_divides(...) { return {}; }

        inline constexpr auto plus = []<class... Args>(Args && ... args) -> decltype(auto) {
            if constexpr (std::is_same_v<decltype(fn_plus(std::forward<Args>(args)...)), undefined>)
                return (... + std::forward<Args>(args));
            else
                return fn_plus(std::forward<Args>(args)...);
        };

        inline constexpr auto multiplies = []<class... Args>(Args && ... args) -> decltype(auto) {
            if constexpr (std::is_same_v<decltype(fn_multiplies(std::forward<Args>(args)...)), undefined>)
                return (... * std::forward<Args>(args));
            else
                return fn_multiplies(std::forward<Args>(args)...);
        };

        inline constexpr auto divides = []<class... Args>(Args && ... args) -> decltype(auto) {
            if constexpr (std::is_same_v<decltype(fn_divides(std::forward<Args>(args)...)), undefined>)
                return (... / std::forward<Args>(args));
            else
                return fn_divides(std::forward<Args>(args)...);
        };

        inline constexpr auto minus = []<class Arg, class... Args>(Arg && arg, Args &&... args) -> decltype(auto) {
            if constexpr (std::is_same_v<decltype(fn_minus(std::forward<Arg>(arg), std::forward<Args>(args)...)),
                              undefined>) {
                if constexpr (sizeof...(Args) == 0)
                    return -std::forward<Arg>(arg);
                else
                    return (std::forward<Arg>(arg) - ... - std::forward<Args>(args));
            } else {
                return fn_minus(std::forward<Arg>(arg), std::forward<Args>(args)...);
            }
        };
    } // namespace arith_impl_
    using arith_impl_::divides;
    using arith_impl_::minus;
    using arith_impl_::multiplies;
    using arith_impl_::plus;
} // namespace gridtools::fn

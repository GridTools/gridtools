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

#include <type_traits>
#include <utility>

namespace gridtools::fn {
    /*
     *   p, f, b, i
     *
     *   p -> p
     *   f -> f
     *   b -> b
     *   i ->
     *   f, p -> f
     *   b, p -> b
     *   f, b -> i
     *   i,
     */
    namespace column_impl_ {
        size_t get_column_size();
        set_column_size(size_t);

        template <class Ptr, class Stride>
        struct column {
            Ptr ptr;
            Stride stride;
        };

        template <class... Ts>
        using has_columns = std::true_type;

        auto make_columnwise(auto fun) {
            return [fun = std::move(fun)]<class... Args>(Args && ... args) {
                if constexpr (!has_columns<std::decay_t<Args>...>())
                    return fun(args...);
            }
        }
        //        inline constexpr auto column = []
        //
    } // namespace column_impl_
} // namespace gridtools::fn

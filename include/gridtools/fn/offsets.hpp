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

#include "../common/integral_constant.hpp"
#include "../common/tuple_util.hpp"
#include "../meta.hpp"
#include "deref.hpp"
#include "shift.hpp"

namespace gridtools::fn {
    namespace offsets_impl_ {

        struct undefined {};

        undefined fn_offsets(...);

        template <class T>
        using to_int = integral_constant<int, T::value>;

        inline constexpr auto offsets = [](auto const &it) {
            if constexpr (std::is_same_v<undefined, decltype(fn_offsets(it))>)
                return meta::transform<to_int, meta::make_indices<tuple_util::size<decltype(it)>, std::tuple>>{};
            else
                return fn_offsets(it);
        };

        constexpr auto reduce(auto fun, auto init) {
            return [fun = fun, init = init](auto const &arg, auto const &... args) {
                auto res = init;
                tuple_util::for_each(
                    [&](auto i) {
                        if (i != -1)
                            res = std::apply(
                                fun, std::tuple(res, fn::deref(fn::shift(i)(arg)), fn::deref(fn::shift(i)(args))...));
                    },
                    offsets(arg));
                return res;
            };
        }

    } // namespace offsets_impl_
    using offsets_impl_::offsets;
    using offsets_impl_::reduce;
} // namespace gridtools::fn

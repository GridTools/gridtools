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

        inline constexpr auto offsets = []<class It>(It const &it) -> decltype(auto) {
            if constexpr (std::is_same_v<undefined, decltype(fn_offsets(it))>)
                return std::array<int, tuple_util::size<It>::value>{};
            else
                return fn_offsets(it);
        };

        template <size_t I>
        struct fast_offset : integral_constant<size_t, I> {
            int offset;
            constexpr fast_offset(std::integral_constant<size_t, I>, int offset) : offset(offset) {}
        };

        template <class T>
        constexpr meta::make_indices<tuple_util::size<T>, std::tuple> make_indices(T const &) {
            return {};
        }

        constexpr auto reduce(auto fun, auto init) {
            return [fun = std::move(fun), init = std::move(init)](auto const &arg, auto const &... args) {
                auto res = init;
                tuple_util::for_each(
                    [&](int offset, auto i) {
                        if (offset == -1)
                            return;
                        auto &&s = fn::shift(fast_offset(i, offset));
                        res = fun(res, fn::deref(s(arg)), fn::deref(s(args))...);
                    },
                    offsets(arg),
                    make_indices(offsets(arg)));
                return res;
            };
        }

    } // namespace offsets_impl_
    using offsets_impl_::fast_offset;
    using offsets_impl_::offsets;
    using offsets_impl_::reduce;
} // namespace gridtools::fn

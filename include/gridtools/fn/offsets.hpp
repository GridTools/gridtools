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
        undefined fn_reduce(...);

        inline constexpr auto offsets = []<class It>(It const &it) -> decltype(auto) {
            if constexpr (std::is_same_v<undefined, decltype(fn_offsets(it))>)
                return std::array<int, tuple_util::size<It>::value>{};
            else
                return fn_offsets(it);
        };

        template <class T>
        constexpr meta::make_indices<tuple_util::size<T>, std::tuple> make_indices(T const &) {
            return {};
        }

        constexpr decltype(auto) get_offsets(auto const &arg, auto &&...) { return offsets(arg); }

        template <auto Fun, auto Init>
        constexpr auto reduce = [](auto const &... args) {
            auto res = Init;
            auto const &offsets = get_offsets(args...);
            tuple_util::for_each(
                [&]<class I>(int offset, I) {
                    if (offset != -1)
                        res = Fun(res, deref(shift<I::value>(args))...);
                },
                offsets,
                make_indices(offsets));
            return res;
        };
    } // namespace offsets_impl_
    using offsets_impl_::offsets;
    using offsets_impl_::reduce;
} // namespace gridtools::fn

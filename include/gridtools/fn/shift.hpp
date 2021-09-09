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

namespace gridtools::fn {
    namespace shift_impl_ {

        struct undefined {};

        undefined fn_shift(...);

        consteval auto shift_overload(meta::val<>) {
            return [](auto const &it) -> decltype(auto) { return it; };
        };

        template <auto Offset>
        consteval auto shift_overload(meta::val<Offset>) {
            return [](auto const &it) {
                if constexpr (!std::is_same_v<undefined, decltype(fn_shift(it, meta::constant<Offset>))>)
                    return fn_shift(it, meta::constant<Offset>);
                else
                    return &tuple_util::get<Offset>(it);
            };
        };

        template <auto Offset0, auto Offset1, auto... Offsets>
        consteval auto shift_overload(meta::val<Offset0, Offset1, Offsets...>) {
            return [](auto const &it) {
                if constexpr (!std::is_same_v<undefined,
                                  decltype(fn_shift(it, meta::constant<Offset0, Offset1, Offsets...>))>)
                    return fn_shift(it, meta::constant<Offset0, Offset1, Offsets...>);
                else if constexpr (!std::is_same_v<undefined, decltype(fn_shift(it, meta::constant<Offset0, Offset1>))>)
                    return shift_overload(meta::constant<Offsets...>)(fn_shift(it, meta::constant<Offset0, Offset1>));
                else
                    return shift_overload(meta::constant<Offset1, Offsets...>)(fn_shift(it, meta::constant<Offset0>));
            };
        };

        template <auto... Offsets>
        constexpr auto shift = shift_overload(meta::constant<Offsets...>);
    } // namespace shift_impl_
    using shift_impl_::shift;
} // namespace gridtools::fn

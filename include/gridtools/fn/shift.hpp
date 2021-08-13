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

namespace gridtools::fn {
    namespace shift_impl_ {

        struct undefined {};

        undefined fn_shift(...);

        constexpr auto shift() {
            return [](auto const &it) { return it; };
        };

        template <class... Ts>
        constexpr std::tuple<Ts...> capture_offsets(Ts &&... offsets) {
            return {std::forward<Ts>(offsets)...};
        }

        template <class Offset>
        constexpr auto shift(Offset &&offset) {
            return [offsets = capture_offsets(std::forward<Offset>(offset))](auto const &it) {
                if constexpr (!std::is_same_v<undefined, decltype(fn_shift(it, std::declval<Offset>()))>)
                    return fn_shift(it, std::get<0>(offsets));
                else if constexpr (is_integral_constant<std::decay_t<Offset>>())
                    return &tuple_util::get<Offset::value>(it);
            };
        };

        template <class Offset0, class Offset1, class... Offsets>
        constexpr auto shift(Offset0 &&offset0, Offset1 &&offset1, Offsets &&... offsets) {
            return [offset0 = capture_offsets(std::forward<Offset0>(offset0)),
                       offset1 = capture_offsets(std::forward<Offset1>(offset1)),
                       offsets = capture_offsets(std::forward<Offsets>(offsets)...)](auto const &it) {
                if constexpr (!std::is_same_v<undefined,
                                  decltype(fn_shift(it,
                                      std::declval<Offset0>(),
                                      std::declval<Offset1>(),
                                      std::declval<Offsets>()...))>)
                    return std::apply(
                        [&](auto const &... offsets) {
                            return fn_shift(it, std::get<0>(offset0), std::get<0>(offset1), offsets...);
                        },
                        offsets);
                else if constexpr (!std::is_same_v<undefined,
                                       decltype(fn_shift(it, std::declval<Offset0>(), std::declval<Offset1>()))>)
                    return std::apply([&](auto const &... offsets) { return shift(offsets...); }, offsets)(
                        fn_shift(it, std::get<0>(offset0), std::get<0>(offset1)));
                else {
                    return std::apply([&](auto const &... offsets) { return shift(std::get<0>(offset1), offsets...); },
                        offsets)(fn_shift(it, std::get<0>(offset0)));
                }
            };
        };
    } // namespace shift_impl_
    using shift_impl_::shift;
} // namespace gridtools::fn

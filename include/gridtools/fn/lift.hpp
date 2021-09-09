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
#include <utility>

#include "../meta.hpp"
#include "offsets.hpp"
#include "shift.hpp"

namespace gridtools::fn {
    namespace lift_impl_ {

        template <auto Stencil, class Args>
        struct lifted_iter {
            Args args;
            friend constexpr bool can_deref(lifted_iter const &it) { return true; }
        };

        template <auto Stencil, class Args, template <auto...> class H, auto... Offsets>
        constexpr auto fn_shift(lifted_iter<Stencil, Args> const &it, H<Offsets...>) {
            auto args = tuple_util::transform(fn::shift<Offsets...>, it.args);
            return lifted_iter<Stencil, decltype(args)>{std::move(args)};
        }

        template <auto Stencil, class Args>
        constexpr auto fn_deref(lifted_iter<Stencil, Args> const &it) {
            return std::apply(Stencil, it.args);
        }

        template <auto Stencil, class Arg, class... Args>
        constexpr decltype(auto) fn_offsets(lifted_iter<Stencil, std::tuple<Arg, Args...>> const &it) {
            return fn::offsets(std::get<0>(it.args));
        }

        struct undefined {};

        undefined fn_ilift(...) { return {}; }

        template <auto Stencil, class... Args>
        auto do_ilift(Args &&... args) {
            if constexpr (std::is_same_v<decltype(fn_ilift(meta::constant<Stencil>, std::forward<Args>(args)...)),
                              undefined>)
                return lifted_iter<Stencil, std::tuple<std::remove_reference_t<Args>...>>{
                    std::tuple(std::forward<Args>(args)...)};
            else
                return fn_ilift(meta::constant<Stencil>, std::forward<Args>(args)...);
        }

        template <auto Stencil>
        constexpr auto ilift = [](auto... its) {
            using res_t = decltype(Stencil(its...));
            if constexpr (meta::is_instantiation_of<std::tuple, res_t>())

                return tuple_util::transform(
                    [&](auto i) {
                        constexpr auto I = decltype(i)::value;
                        constexpr auto f = [](auto const &... its) { return std::get<I>(Stencil(its...)); };
                        return do_ilift<f>(its...);
                    },
                    meta::make_indices<std::tuple_size<res_t>, std::tuple>());
            else
                return do_ilift<Stencil>(std::move(its...));
        };

        template <auto Stencil>
        constexpr auto tlift = [](auto... its) {
            using res_t = decltype(Stencil(its...));
            if constexpr (meta::is_instantiation_of<std::tuple, res_t>())
                return tuple_util::transform(
                    [&](auto i) {
                        constexpr auto I = decltype(i)::value;
                        constexpr auto f = [](auto const &... its) { return std::get<I>(Stencil(its...)); };
                        return fn_tlift(meta::constant<f>, its...);
                    },
                    meta::make_indices<std::tuple_size<res_t>, std::tuple>());
            else
                return fn_tlift(meta::constant<Stencil>, std::move(its...));
        };

        template <auto Stencil, bool UseTmp>
        consteval auto select_lift() {
            if constexpr (UseTmp)
                return tlift<Stencil>;
            else
                return ilift<Stencil>;
        }

        template <auto Stencil, bool UseTmp = false>
        constexpr auto lift = select_lift<Stencil, UseTmp>();

    } // namespace lift_impl_
    using lift_impl_::ilift;
    using lift_impl_::lift;
    using lift_impl_::tlift;
} // namespace gridtools::fn

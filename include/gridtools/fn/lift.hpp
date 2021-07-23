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

#include "offsets.hpp"
#include "shift.hpp"

#include "../meta.hpp"

namespace gridtools::fn {
    namespace lift_impl_ {

        template <class Stencil, class Args>
        struct lifted_iter {
            Stencil stencil;
            Args args;
            constexpr lifted_iter(Stencil stencil, Args args) : stencil(std::move(stencil)), args(std::move(args)) {}
            friend constexpr bool can_deref(lifted_iter const &it) { return true; }
        };

        template <class Stencil, class Args, class... Offsets>
        constexpr auto fn_shift(lifted_iter<Stencil, Args> const &it, Offsets... offsets) {
            return lifted_iter(it.stencil, tuple_util::transform(fn::shift(offsets...), it.args));
        }

        template <class Stencil, class Args>
        constexpr auto fn_deref(lifted_iter<Stencil, Args> const &it) {
            return std::apply(it.stencil, it.args);
        }

        template <class Stencil, class Arg, class... Args>
        constexpr auto fn_offsets(lifted_iter<Stencil, std::tuple<Arg, Args...>> const &it) {
            return fn::offsets(std::get<0>(it.args));
        }

        constexpr auto lift = [](auto stencil) {
            return [stencil = stencil](auto... its) {
                using res_t = decltype(stencil(its...));
                if constexpr (meta::is_instantiation_of<std::tuple, res_t>())
                    return tuple_util::transform(
                        [&](auto i) {
                            return lifted_iter(
                                [&](auto const &... its) { return std::get<decltype(i)::value>(stencil(its...)); },
                                std::tuple(its...));
                        },
                        meta::make_indices<std::tuple_size<res_t>, std::tuple>());
                else
                    return lifted_iter(std::move(stencil), std::tuple(its...));
            };
        };
    } // namespace lift_impl_
    using lift_impl_::lift;
} // namespace gridtools::fn

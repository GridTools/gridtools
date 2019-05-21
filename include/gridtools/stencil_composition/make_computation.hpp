/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <type_traits>
#include <utility>

#include "../common/defs.hpp"
#include "../common/split_args.hpp"
#include "../meta/defs.hpp"
#include "../meta/transform.hpp"
#include "../meta/type_traits.hpp"
#include "computation.hpp"
#include "expandable_parameters/expand_factor.hpp"
#include "expandable_parameters/intermediate_expand.hpp"
#include "grid.hpp"
#include "intermediate.hpp"

namespace gridtools {
    namespace _impl {

#if GT_BROKEN_TEMPLATE_ALIASES
        template <class List>
        struct decay_elements : meta::transform<std::decay, List> {};
#else
        template <class List>
        using decay_elements = meta::transform<decay_t, List>;
#endif

        template <bool IsStateful, class Backend>
        struct make_intermediate_f {
            template <class Grid,
                class... Args,
                class ArgsPair = decltype(
                    split_args<is_arg_storage_pair>(wstd::forward<Args>(std::declval<Args>())...)),
                class ArgStoragePairs = decay_elements<typename ArgsPair::first_type>,
                class Msses = decay_elements<typename ArgsPair::second_type>>
            intermediate<IsStateful, Backend, Grid, ArgStoragePairs, Msses> operator()(
                Grid const &grid, Args &&... args) const {
                // split arg_storage_pair and mss descriptor arguments and forward it to intermediate constructor
                auto &&args_pair = split_args<is_arg_storage_pair>(wstd::forward<Args>(args)...);
                return {grid, wstd::move(args_pair.first)};
            }
        };

    } // namespace _impl

#ifndef NDEBUG
#define GT_POSITIONAL_WHEN_DEBUGGING true
#else
#define GT_POSITIONAL_WHEN_DEBUGGING false
#endif

    /// generator for intermediate
    template <class Backend, class Grid, class Arg, class... Args, enable_if_t<is_grid<Grid>::value, int> = 0>
    auto make_computation(Grid const &grid, Arg &&arg, Args &&... args) {
        return _impl::make_intermediate_f<GT_POSITIONAL_WHEN_DEBUGGING, Backend>{}(
            grid, wstd::forward<Arg>(arg), wstd::forward<Args>(args)...);
    }

#undef GT_POSITIONAL_WHEN_DEBUGGING

    template <class Backend, class Grid, class Arg, class... Args, enable_if_t<is_grid<Grid>::value, int> = 0>
    auto make_positional_computation(Grid const &grid, Arg &&arg, Args &&... args) {
        return _impl::make_intermediate_f<true, Backend>{}(grid, wstd::forward<Arg>(arg), wstd::forward<Args>(args)...);
    }

    // user protection only, catch the case where no backend is specified
    template <class... Args>
    computation<> make_computation(Args &&...) {
        GT_STATIC_ASSERT(!sizeof...(Args), "No backend was specified on a call to make_computation");
        return {};
    }
    template <class... Args>
    computation<> make_positional_computation(Args &&...) {
        GT_STATIC_ASSERT(!sizeof...(Args), "No backend was specified on a call to make_positional_computation");
        return {};
    }
} // namespace gridtools

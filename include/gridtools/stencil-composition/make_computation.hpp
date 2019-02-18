/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
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
                class ArgsPair = decltype(split_args<is_arg_storage_pair>(std::forward<Args>(std::declval<Args>())...)),
                class ArgStoragePairs = GT_META_CALL(decay_elements, typename ArgsPair::first_type),
                class Msses = GT_META_CALL(decay_elements, typename ArgsPair::second_type)>
            intermediate<IsStateful, Backend, Grid, ArgStoragePairs, Msses> operator()(
                Grid const &grid, Args &&... args) const {
                // split arg_storage_pair and mss descriptor arguments and forward it to intermediate constructor
                auto &&args_pair = split_args<is_arg_storage_pair>(std::forward<Args>(args)...);
                return {grid, std::move(args_pair.first), std::move(args_pair.second)};
            }
        };

        template <uint_t Factor, bool IsStateful, class Backend>
        struct make_intermediate_expand_f {
            template <class Grid,
                class... Args,
                class ArgsPair = decltype(split_args<is_arg_storage_pair>(std::forward<Args>(std::declval<Args>())...)),
                class ArgStoragePairs = GT_META_CALL(decay_elements, typename ArgsPair::first_type),
                class Msses = GT_META_CALL(decay_elements, typename ArgsPair::second_type)>
            intermediate_expand<Factor, IsStateful, Backend, Grid, ArgStoragePairs, Msses> operator()(
                Grid const &grid, Args &&... args) const {
                // split arg_storage_pair and mss descriptor arguments and forward it to intermediate constructor
                auto &&args_pair = split_args<is_arg_storage_pair>(std::forward<Args>(args)...);
                return {grid, std::move(args_pair.first), std::move(args_pair.second)};
            }
        };

        /// Dispatch between `intermediate` and `intermediate_expand` on the first parameter type.
        ///
        template <bool Positional, class Backend, class Grid, class... Args, enable_if_t<is_grid<Grid>::value, int> = 0>
        auto make_computation_dispatch(Grid const &grid, Args &&... args)
            GT_AUTO_RETURN((make_intermediate_f<Positional, Backend>{}(grid, std::forward<Args>(args)...)));

        template <bool Positional,
            class Backend,
            class ExpandFactor,
            class Grid,
            class... Args,
            enable_if_t<is_expand_factor<ExpandFactor>::value, int> = 0>
        auto make_computation_dispatch(ExpandFactor, Grid const &grid, Args &&... args) GT_AUTO_RETURN((
            make_intermediate_expand_f<ExpandFactor::value, Positional, Backend>{}(grid, std::forward<Args>(args)...)));

        // user protections
        template <bool,
            class,
            class Arg,
            class... Args,
            enable_if_t<!is_grid<Arg>::value && !is_expand_factor<Arg>::value, int> = 0>
        void make_computation_dispatch(Arg const &, Args &&...) {
            GT_STATIC_ASSERT(sizeof...(Args) < 0, "The computation is malformed");
        }
    } // namespace _impl

#ifndef NDEBUG
#define GT_POSITIONAL_WHEN_DEBUGGING true
#else
#define GT_POSITIONAL_WHEN_DEBUGGING false
#endif

    /// generator for intermediate/intermediate_expand
    ///
    template <class Backend, class Arg, class... Args>
    auto make_computation(Arg const &arg, Args &&... args) GT_AUTO_RETURN(
        (_impl::make_computation_dispatch<GT_POSITIONAL_WHEN_DEBUGGING, Backend>(arg, std::forward<Args>(args)...)));

#undef GT_POSITIONAL_WHEN_DEBUGGING

    template <class Backend, class Arg, class... Args>
    auto make_positional_computation(Arg const &arg, Args &&... args)
        GT_AUTO_RETURN((_impl::make_computation_dispatch<true, Backend>(arg, std::forward<Args>(args)...)));
} // namespace gridtools

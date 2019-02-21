/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
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

    } // namespace _impl

#ifndef NDEBUG
#define GT_POSITIONAL_WHEN_DEBUGGING true
#else
#define GT_POSITIONAL_WHEN_DEBUGGING false
#endif

    /// generator for intermediate
    template <class Backend, class Grid, class Arg, class... Args, enable_if_t<is_grid<Grid>::value, int> = 0>
    auto make_computation(Grid const &grid, Arg &&arg, Args &&... args)
        GT_AUTO_RETURN((_impl::make_intermediate_f<GT_POSITIONAL_WHEN_DEBUGGING, Backend>{}(
            grid, std::forward<Arg>(arg), std::forward<Args>(args)...)));

#undef GT_POSITIONAL_WHEN_DEBUGGING

    template <class Backend, class Grid, class Arg, class... Args, enable_if_t<is_grid<Grid>::value, int> = 0>
    auto make_positional_computation(Grid const &grid, Arg &&arg, Args &&... args) GT_AUTO_RETURN(
        (_impl::make_intermediate_f<true, Backend>{}(grid, std::forward<Arg>(arg), std::forward<Args>(args)...)));

    // user protection only, catch the case where no backend is specified
    template <class... Args>
    computation<> make_computation(Args &&...) {
        GT_STATIC_ASSERT(!sizeof...(Args), "No backend was specified on a call to make_computation");
    }
    template <class... Args>
    computation<> make_positional_computation(Args &&...) {
        GT_STATIC_ASSERT(!sizeof...(Args), "No backend was specified on a call to make_positional_computation");
    }
} // namespace gridtools

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
#include "../common/generic_metafunctions/meta.hpp"
#include "grid.hpp"
#include "expandable_parameters/expand_factor.hpp"
#include "expandable_parameters/intermediate_expand.hpp"
#include "intermediate.hpp"

namespace gridtools {
    namespace _impl {

        template < class List >
        using decay_elements = meta::apply< meta::transform< meta::meta_t_< std::decay >::apply >, List >;

        template < template < uint_t, bool, class, class, class, class > class Intermediate,
            uint_t Factor,
            bool IsStateful,
            class Backend,
            class Grid,
            class... Args,
            class ArgsPair = decltype(
                split_args< is_arg_storage_pair >(std::forward< Args >(std::declval< Args >())...)),
            class ArgStoragePairs = decay_elements< meta::first< ArgsPair > >,
            class Msses = decay_elements< meta::second< ArgsPair > > >
        Intermediate< Factor, IsStateful, Backend, Grid, ArgStoragePairs, Msses > make_intermediate(
            Grid const &grid, Args &&... args) {
            auto &&args_pair = split_args< is_arg_storage_pair >(std::forward< Args >(args)...);
            return {grid, std::move(args_pair.first), std::move(args_pair.second)};
        }

        template < bool Positional,
            class Backend,
            class Grid,
            class... Args,
            typename std::enable_if< is_grid< Grid >::value, int >::type = 0 >
        auto make_computation_dispatch(Grid const &grid, Args &&... args) GT_AUTO_RETURN(
            (make_intermediate< intermediate, 1, Positional, Backend >(grid, std::forward< Args >(args)...)));

        template < bool Positional,
            class Backend,
            class ExpandFactor,
            class Grid,
            class... Args,
            typename std::enable_if< is_expand_factor< ExpandFactor >::value, int >::type = 0 >
        auto make_computation_dispatch(ExpandFactor, Grid const &grid, Args &&... args)
            GT_AUTO_RETURN((make_intermediate< intermediate_expand, ExpandFactor::value, Positional, Backend >(
                grid, std::forward< Args >(args)...)));

        // user protections
        template < bool, class, class... Args >
        void make_computation_dispatch(Args...) {
            GRIDTOOLS_STATIC_ASSERT(sizeof...(Args) < 0, "The computation is malformed");
        }
    }

#ifndef NDEBUG
#define POSITIONAL_WHEN_DEBUGGING true
#ifndef SUPPRESS_MESSAGES
#pragma message( \
    ">>\n>> In debug mode each computation is positional,\n>> so the loop indices can be queried from within\n>> the operator functions")
#endif
#else
#define POSITIONAL_WHEN_DEBUGGING false
#endif

    template < class Backend, class... Args >
    auto make_computation(Args &&... args) GT_AUTO_RETURN(
        (_impl::make_computation_dispatch< POSITIONAL_WHEN_DEBUGGING, Backend >(std::forward< Args >(args)...)));

#undef POSITIONAL_WHEN_DEBUGGING

    template < class Backend, class... Args >
    auto make_positional_computation(Args &&... args)
        GT_AUTO_RETURN((_impl::make_computation_dispatch< true, Backend >(std::forward< Args >(args)...)));
}

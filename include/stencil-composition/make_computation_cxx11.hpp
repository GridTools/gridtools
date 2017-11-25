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
#include <memory>
#include <type_traits>
#include <utility>

#include <boost/fusion/include/make_vector.hpp>

#include "../common/defs.hpp"
#include "expandable_parameters/expand_factor.hpp"
#include "expandable_parameters/intermediate_expand.hpp"
#include "intermediate.hpp"

namespace gridtools {
    namespace _impl {
        template < bool Positional,
            typename Backend,
            typename Domain,
            typename Grid,
            typename... MssDescriptorTrees,
            typename Res = intermediate< Backend,
                boost::fusion::vector< typename std::decay< MssDescriptorTrees >::type... >,
                typename std::decay< Domain >::type,
                Grid,
                Positional > >
        std::shared_ptr< Res > make_computation(
            Domain &&domain, const Grid &grid, MssDescriptorTrees &&... mss_descriptor_trees) {
            return std::make_shared< Res >(std::forward< Domain >(domain),
                grid,
                boost::fusion::make_vector(std::forward< MssDescriptorTrees >(mss_descriptor_trees)...));
        }

        template < bool Positional,
            typename Backend,
            typename Expand,
            typename Domain,
            typename Grid,
            typename... MssDescriptorTrees,
            typename Res = intermediate_expand< Backend,
                boost::fusion::vector< typename std::decay< MssDescriptorTrees >::type... >,
                typename std::decay< Domain >::type,
                Grid,
                Positional,
                Expand > >
        std::shared_ptr< Res > make_computation_expandable(
            Domain &&domain, const Grid &grid, MssDescriptorTrees &&... mss_descriptor_trees) {
            return std::make_shared< Res >(std::forward< Domain >(domain),
                grid,
                boost::fusion::make_vector(std::forward< MssDescriptorTrees >(mss_descriptor_trees)...));
        }

        template < bool Positional,
            typename Backend,
            typename Arg,
            typename... Args,
            typename = typename std::enable_if< is_aggregator_type< typename std::decay< Arg >::type >::value >::type >
        auto make_computation_proxy(Arg &&arg, Args &&... args) GT_AUTO_RETURN(
            (make_computation< Positional, Backend >(std::forward< Arg >(arg), std::forward< Args >(args)...)));

        template < bool Positional,
            typename Backend,
            typename Arg,
            typename... Args,
            typename = typename std::enable_if< is_expand_factor< Arg >::value >::type >
        auto make_computation_proxy(Arg arg, Args &&... args)
            GT_AUTO_RETURN((make_computation_expandable< Positional, Backend, Arg >(std::forward< Args >(args)...)));

        // user protections
        template < bool, typename, typename... Args >
        short_t make_computation_dispatch(Args...) {
            GRIDTOOLS_STATIC_ASSERT((sizeof...(Args)), "The computation is malformed");
            return -1;
        }
    }

    template < typename Backend, typename... Args >
    auto make_computation(Args &&... args) GT_AUTO_RETURN(
        (_impl::make_computation_dispatch< POSITIONAL_WHEN_DEBUGGING, Backend >(std::forward< Args >(args)...)));

    template < typename Backend, typename... Args >
    auto make_positional_computation(Args &&... args)
        GT_AUTO_RETURN((_impl::make_computation_dispatch< true, Backend >(std::forward< Args >(args)...)));
}

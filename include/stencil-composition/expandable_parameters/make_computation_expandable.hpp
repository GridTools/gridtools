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

#include "../conditionals/fill_conditionals.hpp"
#include "../../common/generic_metafunctions/vector_to_set.hpp"
#include "../computation_grammar.hpp"
#include "expand_factor.hpp"
#include "intermediate_expand.hpp"

/**
   @file make_computation specific for expandable parameters
*/

namespace gridtools {

    template < bool Positional,
        typename Backend,
        typename ReductionType,
        typename Expand,
        typename Domain,
        typename Grid,
        typename... Mss >
    std::shared_ptr< computation< ReductionType > > make_computation_expandable_impl(
        Expand /**/, Domain &domain, const Grid &grid, Mss... args_) {

        // doing type checks and defining the conditionals set
        typedef typename _impl::create_conditionals_set< Domain, Grid, Mss... >::type conditionals_set_t;

        conditionals_set_t conditionals_set_;

        fill_conditionals(conditionals_set_, args_...);

        return std::make_shared< intermediate_expand< Backend,
            meta_array< typename meta_array_generator< boost::mpl::vector0<>, Mss... >::type,
                                                          boost::mpl::quote1< is_computation_token > >,
            Domain,
            Grid,
            conditionals_set_t,
            ReductionType,
            Positional,
            Expand > >(domain, grid, conditionals_set_);
    }

    template < typename Backend,
        typename Expand,
        typename Domain,
        typename Grid,
        typename... Mss,
        typename = typename std::enable_if< is_expand_factor< Expand >::value >::type >
    std::shared_ptr< computation< typename _impl::reduction_helper< Mss... >::reduction_type_t > > make_computation(
        Expand /**/, Domain &domain, const Grid &grid, Mss... args_) {
        GRIDTOOLS_STATIC_ASSERT(is_expand_factor< Expand >::value, "type error");
        return make_computation_expandable_impl< POSITIONAL_WHEN_DEBUGGING,
            Backend,
            typename _impl::reduction_helper< Mss... >::reduction_type_t >(Expand(), domain, grid, args_...);
    }

    template < typename Backend,
        typename Expand,
        typename Domain,
        typename Grid,
        typename... Mss,
        typename = typename std::enable_if< is_expand_factor< Expand >::value >::type >
    std::shared_ptr< computation< typename _impl::reduction_helper< Mss... >::reduction_type_t > >
    make_positional_computation(Expand /**/, Domain &domain, const Grid &grid, Mss... args_) {
        return make_computation_expandable_impl< true, Backend >(Expand(), domain, grid, args_...);
    }
}

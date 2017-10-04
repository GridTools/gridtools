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
#include "../defs.hpp"

namespace gridtools {
    namespace impl {
        template < uint_t cnt, typename Value, uint_t Threshold, typename VariadicHolder, Value... Rest >
        struct shorten_impl;

        template < uint_t cnt, typename Value, Value... Args, template < Value... > class VariadicHolder >
        struct shorten_impl< cnt, Value, cnt, VariadicHolder< Args... > > {
            using type = VariadicHolder< Args... >;
        };

        template < uint_t cnt,
            typename Value,
            uint_t Threshold,
            Value... Args,
            Value FirstRest,
            Value... Rest,
            template < Value... > class VariadicHolder >
        struct shorten_impl< cnt, Value, Threshold, VariadicHolder< Args... >, FirstRest, Rest... > {
            using type = typename boost::mpl::eval_if_c< cnt == Threshold,
                boost::mpl::identity< VariadicHolder< Args... > >,
                shorten_impl< cnt + 1, Value, Threshold, VariadicHolder< Args..., FirstRest >, Rest... > >::type;
        };
    }

    /**
     * @struct shorten
     * Given a type with a set of variadic templates, returns the same type with only the
     * first "Threshold" number of variadic templates. Threshold has to be smaller or equal than
     * the number of variadic templates contained in the holder type
     * Example of use:
     *   shorten<int, vector<3,4,5>, 2> == vector<3,4>
     */
    template < typename Value, typename VariadicHolder, uint_t Threshold >
    struct shorten;

    template < typename Value,
        Value First,
        Value... Args,
        template < Value... > class VariadicHolder,
        uint_t Threshold >
    struct shorten< Value, VariadicHolder< First, Args... >, Threshold > {
        GRIDTOOLS_STATIC_ASSERT((Threshold <= sizeof...(Args) + 1), GT_INTERNAL_ERROR);
        using type = typename impl::shorten_impl< 0, Value, Threshold, VariadicHolder<>, First, Args... >::type;
    };
}

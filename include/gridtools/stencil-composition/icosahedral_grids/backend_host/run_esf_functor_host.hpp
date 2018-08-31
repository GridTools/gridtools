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

#include "../../../common/defs.hpp"
#include "../../../common/generic_metafunctions/type_traits.hpp"
#include "../../functor_decorator.hpp"
#include "../../run_esf_functor.hpp"
#include "../../run_functor_arguments.hpp"
#include "../iterate_domain_remapper.hpp"

namespace gridtools {

    /*
     * @brief main functor that executes (for host) the user functor of an ESF
     * @tparam RunFunctorArguments run functor arguments
     * @tparam Interval interval where the functor gets executed
     */
    struct run_esf_functor_host {
        template <class EsfArguments,
            class EsfColor = typename EsfArguments::esf_t::color_t,
            class Color = typename EsfArguments::color_t>
        struct color_esf_match
            : bool_constant<std::is_same<EsfColor, Color>::value || std::is_same<EsfColor, nocolor>::value> {};

        template <class IntervalType,
            class EsfArguments,
            class ItDomain,
            enable_if_t<color_esf_match<EsfArguments>::value, int> = 0>
        GT_FUNCTION void operator()(ItDomain &it_domain) const {
            GRIDTOOLS_STATIC_ASSERT((is_esf_arguments<EsfArguments>::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT(
                !EsfArguments::is_reduction_t::value, "Reductions not supported at the moment for icosahedral grids");

            typedef typename EsfArguments::functor_t original_functor_t;
            typedef typename EsfArguments::esf_t esf_t;
            typedef typename esf_t::template esf_function<EsfArguments::color_t::color_t::value> colored_functor_t;

            typedef functor_decorator<typename original_functor_t::id,
                colored_functor_t,
                typename original_functor_t::repeat_t,
                IntervalType>
                functor_t;

            GRIDTOOLS_STATIC_ASSERT(is_functor_decorator<functor_t>::value, GT_INTERNAL_ERROR);

            typedef typename get_trivial_iterate_domain_remapper<ItDomain,
                typename EsfArguments::esf_t,
                typename EsfArguments::color_t>::type iterate_domain_remapper_t;

            iterate_domain_remapper_t iterate_domain_remapper(it_domain);

            call_repeated<functor_t, IntervalType>(iterate_domain_remapper);
        }

        template <class IntervalType,
            class EsfArguments,
            class ItDomain,
            enable_if_t<!color_esf_match<EsfArguments>::value, int> = 0>
        GT_FUNCTION void operator()(ItDomain &) const {}
    };
} // namespace gridtools

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

#include <boost/mpl/has_key.hpp>

#include "../../../common/generic_metafunctions/type_traits.hpp"
#include "../../functor_decorator.hpp"
#include "../../run_esf_functor.hpp"
#include "../../run_functor_arguments.hpp"
#include "../iterate_domain_remapper.hpp"

namespace gridtools {

    /*
     * @brief main functor that executes (for mic) the user functor of an ESF
     * @tparam RunFunctorArguments run functor arguments
     * @tparam Interval interval where the functor gets executed
     */
    template <typename RunFunctorArguments, typename Interval>
    struct run_esf_functor_mic {
        GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArguments>::value), GT_INTERNAL_ERROR);

        template <class Index, class ItDomain>
        GT_FUNCTION void operator()(Index, ItDomain &it_domain) const {

            typedef esf_arguments<RunFunctorArguments, Index> esf_arguments_t;
            typedef typename esf_arguments_t::interval_map_t interval_map_t;

            if (boost::mpl::has_key<interval_map_t, Interval>::value) {
                typedef typename boost::mpl::at<interval_map_t, Interval>::type interval_type;
                do_impl<interval_type, esf_arguments_t>(it_domain);
            }
        }
        /*
         * @brief main functor implemenation that executes (for Mic) the user functor of an ESF
         *      (specialization for non reduction operations)
         * @tparam IntervalType interval where the functor gets executed
         * @tparam EsfArgument esf arguments type that contains the arguments needed to execute this ESF.
         */
        template <class IntervalType,
            class EsfArguments,
            class ItDomain,
            enable_if_t<!EsfArguments::is_reduction_t::value, int> = 0>
        GT_FUNCTION void do_impl(ItDomain &it_domain) const {
            GRIDTOOLS_STATIC_ASSERT((is_esf_arguments<EsfArguments>::value), GT_INTERNAL_ERROR);
            using functor_t = typename EsfArguments::functor_t;

            GRIDTOOLS_STATIC_ASSERT(is_functor_decorator<functor_t>::value, GT_INTERNAL_ERROR);

            using iterate_domain_remapper_t =
                typename get_iterate_domain_remapper<ItDomain, typename EsfArguments::esf_t::args_t>::type;
            iterate_domain_remapper_t iterate_domain_remapper(it_domain);

            call_repeated<functor_t, IntervalType>(iterate_domain_remapper);
        }

        /*
         * @brief main functor implemenation that executes (for Mic) the user functor of an ESF
         *      (specialization for reduction operations)
         * @tparam IntervalType interval where the functor gets executed
         * @tparam EsfArgument esf arguments type that contains the arguments needed to execute this ESF.

         TODO: reduction at the current state will not work with expandable parameters and default interval
         */
        template <typename IntervalType,
            typename EsfArguments,
            class ItDomain,
            enable_if_t<EsfArguments::is_reduction_t::value, int> = 0>
        GT_FUNCTION void do_impl(ItDomain &it_domain) const {
            using functor_t = typename EsfArguments::functor_t;
            using bin_op_t = typename EsfArguments::reduction_data_t::bin_op_t;
            GRIDTOOLS_STATIC_ASSERT((is_esf_arguments<EsfArguments>::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((functor_t::repeat_t::value == 1),
                "Expandable parameters are not implemented for the reduction stages");
            using iterate_domain_remapper_t =
                typename get_iterate_domain_remapper<ItDomain, typename EsfArguments::esf_t::args_t>::type;
            iterate_domain_remapper_t iterate_domain_remapper(it_domain);
            it_domain.set_reduction_value(bin_op_t{}(
                it_domain.reduction_value(), functor_t::f_type::Do(iterate_domain_remapper, IntervalType{})));
        }
    };
} // namespace gridtools

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

#include <boost/mpl/at.hpp>

#include "../../functor_decorator.hpp"
#include "../../run_esf_functor.hpp"
#include "../../run_functor_arguments.hpp"
#include "../iterate_domain_remapper.hpp"

namespace gridtools {
    /*
     * @brief main functor that executes (for CUDA) the user functor of an ESF
     * @tparam RunFunctorArguments run functor arguments
     * @tparam Interval interval where the functor gets executed
     */
    struct run_esf_functor_cuda {
        /*
         * @brief main functor implemenation that executes (for CUDA) the user functor of an ESF
         * @tparam IntervalType interval where the functor gets executed
         * @tparam EsfArgument esf arguments type that contains the arguments needed to execute this ESF.
         */
        template <class IntervalType, class EsfArguments, class ItDomain>
        GT_FUNCTION_DEVICE void operator()(ItDomain &it_domain) const {
            GRIDTOOLS_STATIC_ASSERT((is_esf_arguments<EsfArguments>::value), GT_INTERNAL_ERROR);

            typedef typename EsfArguments::extent_t extent_t;
            typedef typename EsfArguments::functor_t functor_t;

            // a grid point at the core of the block can be out of extent (for last blocks) if domain of computations
            // is not a multiple of the block size
            if (it_domain.template is_thread_in_domain<extent_t>()) {
                // instantiate the iterate domain remapper, that will map the calls to arguments to their actual
                // position in the iterate domain
                typedef typename get_iterate_domain_remapper<ItDomain, typename EsfArguments::esf_args_map_t>::type
                    iterate_domain_remapper_t;

                iterate_domain_remapper_t iterate_domain_remapper(it_domain);

                // call the user functor at the core of the block
                call_repeated<functor_t, IntervalType>(iterate_domain_remapper);
            }

            // synchronize threads if not independent esf
            if (!boost::mpl::at<typename EsfArguments::async_esf_map_t, functor_t>::type::value)
                __syncthreads();
        }
    };
} // namespace gridtools

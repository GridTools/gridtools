/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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
#include <boost/utility/enable_if.hpp>
#include "../../run_esf_functor.hpp"
#include "../../block_size.hpp"
#include "../iterate_domain_remapper.hpp"

namespace gridtools {
    /*
     * @brief main functor that executes (for CUDA) the user functor of an ESF
     * @tparam RunFunctorArguments run functor arguments
     * @tparam Interval interval where the functor gets executed
     */
    template < typename RunFunctorArguments, typename Interval >
    struct run_esf_functor_cuda
        : public run_esf_functor< run_esf_functor_cuda< RunFunctorArguments, Interval > > // CRTP
    {
        GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments< RunFunctorArguments >::value), "Internal Error: wrong type");
        // TODOCOSUNA This type here is not an interval, is a pair<int_, int_ >
        // BOOST_STATIC_ASSERT((is_interval<Interval>::value));

        typedef run_esf_functor< run_esf_functor_cuda< RunFunctorArguments, Interval > > super;
        typedef typename RunFunctorArguments::physical_domain_block_size_t physical_domain_block_size_t;
        typedef typename RunFunctorArguments::processing_elements_block_size_t processing_elements_block_size_t;

        // metavalue that determines if a warp is processing more grid points that the default assigned
        // at the core of the block
        typedef typename boost::mpl::not_< typename boost::is_same< physical_domain_block_size_t,
            processing_elements_block_size_t >::type >::type multiple_grid_points_per_warp_t;

        // nevertheless, even if each thread computes more than a grid point, the i size of the physical block
        // size and the cuda block size have to be the same
        GRIDTOOLS_STATIC_ASSERT(
            (physical_domain_block_size_t::i_size_t::value == processing_elements_block_size_t::i_size_t::value),
            "Internal Error: wrong type");

        typedef typename RunFunctorArguments::iterate_domain_t iterate_domain_t;

        GT_FUNCTION
        explicit run_esf_functor_cuda(iterate_domain_t &iterate_domain) : super(iterate_domain) {}

        /*
         * @brief main functor implemenation that executes (for CUDA) the user functor of an ESF
         * @tparam IntervalType interval where the functor gets executed
         * @tparam EsfArgument esf arguments type that contains the arguments needed to execute this ESF.
         */
        template < typename IntervalType, typename EsfArguments >
        __device__ static void do_impl(iterate_domain_t &it_domain) {
            GRIDTOOLS_STATIC_ASSERT((is_esf_arguments< EsfArguments >::value), "Internal Error: wrong type");

            // instantiate the iterate domain remapper, that will map the calls to arguments to their actual
            // position in the iterate domain
            typedef typename get_iterate_domain_remapper< iterate_domain_t,
                typename EsfArguments::esf_args_map_t >::type iterate_domain_remapper_t;

            iterate_domain_remapper_t iterate_domain_remapper(it_domain);

            typedef typename EsfArguments::functor_t functor_t;
            typedef typename EsfArguments::extent_t extent_t;

            // a grid point at the core of the block can be out of extent (for last blocks) if domain of computations
            // is not a multiple of the block size
            if (it_domain.template is_thread_in_domain< extent_t >()) {
                // call the user functor at the core of the block (multiple times in case of expandable parameters)
                _impl::call_repeated< functor_t::repeat_t::value, functor_t, iterate_domain_remapper_t, IntervalType >::
                    Do(iterate_domain_remapper);
            }

            // synchronize threads if not independent esf
            if (!boost::mpl::at< typename EsfArguments::async_esf_map_t, functor_t >::type::value)
                __syncthreads();
        }
    };
}

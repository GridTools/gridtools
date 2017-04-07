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
#include <boost/static_assert.hpp>
#include <boost/mpl/for_each.hpp>
#include "basic_token_execution.hpp"
#include "backend_traits_fwd.hpp"
#include "run_functor_arguments_fwd.hpp"

/**
@file Implementation of the k loop execution policy
The policies which are currently considered are
 - forward: the k loop is executed upward, increasing the value of the iterator on k. This is the option to be used when
the stencil operations at level k depend on the fields at level k-1 (forward substitution).
 - backward: the k loop is executed downward, decreasing the value of the iterator on k. This is the option to be used
when the stencil operations at level k depend on the fields at level k+1 (backward substitution).
 - parallel: the operations on each k level are executed in parallel. This is feasable only if there are no dependencies
between levels.
*/
namespace gridtools {

    namespace _impl {

        /**
           @brief   Execution kernel containing the loop over k levels
        */
        template < typename ExecutionEngine, typename ExtraArguments >
        struct run_f_on_interval;

        /**
           @brief partial specialization for the forward or backward cases
        */
        template < enumtype::execution IterationType, typename RunFunctorArguments >
        struct run_f_on_interval< enumtype::execute< IterationType >, RunFunctorArguments >
            : public run_f_on_interval_base<
                  run_f_on_interval< enumtype::execute< IterationType >, RunFunctorArguments > > // CRTP
        {
            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments< RunFunctorArguments >::value), GT_INTERNAL_ERROR);

            typedef
                typename backend_traits_from_id< RunFunctorArguments::backend_ids_t::s_backend_id >::run_esf_functor_h_t
                    run_esf_functor_h_t;
            typedef run_f_on_interval_base<
                run_f_on_interval< typename enumtype::execute< IterationType >, RunFunctorArguments > > super;
            typedef typename super::iterate_domain_t iterate_domain_t;
            typedef typename enumtype::execute< IterationType >::type execution_engine;
            typedef typename RunFunctorArguments::functor_list_t functor_list_t;

            GT_FUNCTION
            explicit run_f_on_interval(
                iterate_domain_t &iterate_domain_, typename RunFunctorArguments::grid_t const &grid)
                : super(iterate_domain_, grid) {}

            template < typename IterationPolicy, typename Interval >
            GT_FUNCTION void k_loop(int_t from, int_t to) const {
#ifdef CUDA8
                assert(from >= 0);
                assert(to >= 0);
                assert(to >= from);
#endif
                typedef typename run_esf_functor_h_t::template apply< RunFunctorArguments, Interval >::type
                    run_esf_functor_t;

                for (int_t k = from; k <= to; ++k, IterationPolicy::increment(super::m_domain)) {
                    boost::mpl::for_each< boost::mpl::range_c< int, 0, boost::mpl::size< functor_list_t >::value > >(
                        run_esf_functor_t(super::m_domain));
                }
            }
        };

        /**
           @brief partial specialization for the parallel case (to be implemented)
           stub
         */
        template < typename RunFunctorArguments >
        struct run_f_on_interval< typename enumtype::execute< enumtype::parallel >, RunFunctorArguments >
            : public run_f_on_interval_base<
                  run_f_on_interval< enumtype::execute< enumtype::parallel >, RunFunctorArguments > > {
            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments< RunFunctorArguments >::value), GT_INTERNAL_ERROR);
            //*TODO implement me
        };
    } // namespace _impl
} // namespace gridtools

/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
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
            GRIDTOOLS_STATIC_ASSERT(
                (is_run_functor_arguments< RunFunctorArguments >::value), "Internal Error: wrong type");

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

            template<typename IterationPolicy, typename Interval>
            GT_FUNCTION
            void k_loop(int_t from, int_t to) const {
                // assert(from>=0); // PRETTY_FUNCTION issue
                // assert(to>=0);
                // assert(to>=from);
                typedef typename run_esf_functor_h_t::template apply<RunFunctorArguments, Interval>::type run_esf_functor_t;

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
            GRIDTOOLS_STATIC_ASSERT(
                (is_run_functor_arguments< RunFunctorArguments >::value), "Internal Error: wrong type");
            //*TODO implement me
        };
    } // namespace _impl
} // namespace gridtools

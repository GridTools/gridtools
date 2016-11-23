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
#include "interval.hpp"
#include "iteration_policy.hpp"
#include "level.hpp"
#include <boost/mpl/has_key.hpp>

namespace gridtools {
    namespace _impl {

        namespace {
            /**
               @brief generic forward declaration of the execution_policy struct.
            */
            template < typename RunF >
            struct run_f_on_interval_esf_arguments;

            template < typename RunF >
            struct run_f_on_interval_run_functor_arguments;

            template < typename RunF >
            struct run_f_on_interval_execution_engine;

            /**
               @brief forward declaration of the execution_policy struct
            */
            template < typename ExecutionEngine,
                typename RunFunctorArguments,
                template < typename, typename > class Impl >
            struct run_f_on_interval_run_functor_arguments< Impl< ExecutionEngine, RunFunctorArguments > > {
                typedef RunFunctorArguments type;
            };
            template < typename ExecutionEngine,
                typename RunFunctorArguments,
                template < typename, typename > class Impl >
            struct run_f_on_interval_execution_engine< Impl< ExecutionEngine, RunFunctorArguments > > {
                typedef ExecutionEngine type;
            };

        } // unnamed namespace

        /**
           @brief basic token of execution responsible of handling the discretization over the vertical dimension. This
           may be done with a loop over k or by partitoning the k axis and executing in parallel, depending on the
           execution_policy defined in the multi-stage stencil. The base class is then specialized using the CRTP
           pattern for the different policies.
        */
        template < typename RunFOnIntervalImpl >
        struct run_f_on_interval_base {
            /**\brief necessary because the Derived class is an incomplete type at the moment of the instantiation of
             * the base class*/
            typedef
                typename run_f_on_interval_run_functor_arguments< RunFOnIntervalImpl >::type run_functor_arguments_t;
            typedef typename run_f_on_interval_execution_engine< RunFOnIntervalImpl >::type execution_engine;

            typedef typename run_functor_arguments_t::local_domain_t local_domain_t;
            typedef typename run_functor_arguments_t::iterate_domain_t iterate_domain_t;
            typedef typename run_functor_arguments_t::grid_t grid_t;

            GT_FUNCTION
            explicit run_f_on_interval_base(iterate_domain_t &domain, grid_t const &grid)
                : m_grid(grid), m_domain(domain) {}

            template < typename Interval >
            GT_FUNCTION void operator()(Interval const &) const {
                typedef typename index_to_level< typename Interval::first >::type from_t;
                typedef typename index_to_level< typename Interval::second >::type to_t;

                // check that the axis specified by the user are containing the k interval
                GRIDTOOLS_STATIC_ASSERT(
                    (level_to_index< typename grid_t::axis_type::FromLevel >::value <= Interval::first::value &&
                        level_to_index< typename grid_t::axis_type::ToLevel >::value >= Interval::second::value),
                    "the k interval exceeds the axis you specified for the grid instance");

                typedef iteration_policy< from_t,
                    to_t,
                    typename grid_traits_from_id< run_functor_arguments_t::backend_ids_t::s_grid_type_id >::dim_k_t,
                    execution_engine::type::iteration >
                    iteration_policy_t;

                uint_t const from = m_grid.template value_at< from_t >();
                uint_t const to = m_grid.template value_at< to_t >();

                static_cast< RunFOnIntervalImpl * >(const_cast< run_f_on_interval_base< RunFOnIntervalImpl > * >(this))
                    ->template k_loop< iteration_policy_t, Interval >(from, to);
            }

          protected:
            grid_t const &m_grid;
            iterate_domain_t &m_domain;
        };

    } // namespace _impl
} // namespace gridtools

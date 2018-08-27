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

/**
@file
Implementation of the k loop execution policy
The policies which are currently considered are
 - forward: the k loop is executed upward, increasing the value of the iterator on k. This is the option to be used when
the stencil operations at level k depend on the fields at level k-1 (forward substitution).
 - backward: the k loop is executed downward, decreasing the value of the iterator on k. This is the option to be used
when the stencil operations at level k depend on the fields at level k+1 (backward substitution).
 - parallel: the operations on each k level are executed in parallel. This is feasable only if there are no dependencies
between levels.
*/

#pragma once

#include "../common/defs.hpp"
#include "../common/generic_metafunctions/for_each.hpp"
#include "../common/generic_metafunctions/meta.hpp"
#include "../common/host_device.hpp"

#include "execution_types.hpp"
#include "grid_traits_fwd.hpp"
#include "interval.hpp"
#include "iteration_policy.hpp"
#include "level.hpp"
#include "loop_interval.hpp"
#include "run_functor_arguments.hpp"

namespace gridtools {

    template <class FromLevel, class ToLevel, class BackendIds, class ExecutionEngine, class Grid>
    GT_FUNCTION int get_k_interval(BackendIds, ExecutionEngine, Grid const &grid);

    namespace _impl {

        /**
           @brief basic token of execution responsible of handling the discretization over the vertical dimension. This
           may be done with a loop over k or by partitoning the k axis and executing in parallel, depending on the
           execution_policy defined in the multi-stage stencil. The base class is then specialized using the CRTP
           pattern for the different policies.
        */
        /**
           @brief   Execution kernel containing the loop over k levels
        */
        template <typename RunFunctorArguments, class RunEsfFunctor, class ItDomain, class Grid>
        struct run_f_on_interval {
            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArguments>::value), GT_INTERNAL_ERROR);

            typedef typename RunFunctorArguments::execution_type_t execution_engine;
            typedef typename RunFunctorArguments::functor_list_t functor_list_t;

            ItDomain &m_domain;
            Grid const &m_grid;

            template <typename IterationPolicy, class Stages>
            GT_FUNCTION void k_loop(int_t from, int_t to) const {
                const bool in_domain =
                    m_domain.template is_thread_in_domain<typename RunFunctorArguments::max_extent_t>();

                if (in_domain)
                    m_domain.template begin_fill<IterationPolicy>();

                for (int_t k = from; k <= to; ++k, IterationPolicy::increment(m_domain)) {
                    if (in_domain) {
                        const int_t lev = (IterationPolicy::value == enumtype::backward) ? (to - k) + from : k;
                        m_domain.template fill_caches<IterationPolicy>(lev, m_grid);
                    }

                    RunEsfFunctor::template exec<Stages>(m_domain);

                    if (in_domain) {

                        const int_t lev = (IterationPolicy::value == enumtype::backward) ? (to - k) + from : k;

                        m_domain.template flush_caches<IterationPolicy>(lev, m_grid);
                        m_domain.template slide_caches<IterationPolicy>();
                    }
                }
                if (in_domain)
                    m_domain.template final_flush<IterationPolicy>();
            }

            template <class From, class To, class Stages>
            GT_FUNCTION void operator()(loop_interval<From, To, Stages>) const {
                using from_index_t = GT_META_CALL(level_to_index, From);
                using to_index_t = GT_META_CALL(level_to_index, To);
                // check that the axis specified by the user are containing the k interval
                GRIDTOOLS_STATIC_ASSERT(
                    (level_to_index<typename Grid::axis_type::FromLevel>::type::value <= from_index_t::value &&
                        level_to_index<typename Grid::axis_type::ToLevel>::type::value >= to_index_t::value),
                    "the k interval exceeds the axis you specified for the grid instance");

                typedef iteration_policy<From, To, execution_engine::iteration> iteration_policy_t;

                const auto k_interval = get_k_interval<From, To>(typename RunFunctorArguments::backend_ids_t{},
                    typename RunFunctorArguments::execution_type_t{},
                    m_grid);

                // for parallel execution we might get empty intervals,
                // for other execution policies we check that they are given in the correct order
                assert(RunFunctorArguments::execution_type_t::iteration == enumtype::parallel ||
                       k_interval.first <= k_interval.second);
                if (k_interval.first <= k_interval.second)
                    k_loop<iteration_policy_t, Stages>(k_interval.first, k_interval.second);
            }
        };
    } // namespace _impl

    template <class RunFunctorArguments, class RunEsfFunctor, class ItDomain, class Grid>
    GT_FUNCTION void run_functors_on_interval(ItDomain &it_domain, Grid const &grid) {
        gridtools::for_each<typename RunFunctorArguments::new_loop_intervals_t>(
            _impl::run_f_on_interval<RunFunctorArguments, RunEsfFunctor, ItDomain, Grid>{it_domain, grid});
    }
} // namespace gridtools

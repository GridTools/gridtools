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

#include "../common/array.hpp"
#include "../common/defs.hpp"
#include "../common/generic_metafunctions/for_each.hpp"
#include "../common/host_device.hpp"
#include "../meta/at.hpp"
#include "../meta/first.hpp"
#include "../meta/last.hpp"
#include "../meta/length.hpp"
#include "../meta/macros.hpp"
#include "../meta/second.hpp"
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
           may be done with a loop over k or by partitioning the k axis and executing in parallel, depending on the
           execution_policy defined in the multi-stage stencil.
        */
        /**
           @brief   Execution kernel containing the loop over k levels
        */
        template <typename RunFunctorArguments, class RunEsfFunctor, class ItDomain, class Grid>
        struct run_f_on_interval_with_k_caches {
            GRIDTOOLS_STATIC_ASSERT(is_run_functor_arguments<RunFunctorArguments>::value, GT_INTERNAL_ERROR);

            using execution_type_t = typename RunFunctorArguments::execution_type_t;
            using loop_intervals_t = typename RunFunctorArguments::loop_intervals_t;
            using first_t = GT_META_CALL(meta::first, loop_intervals_t);
            using last_t = GT_META_CALL(meta::last, loop_intervals_t);

            ItDomain &m_domain;
            bool m_in_domain;
            Grid const &m_grid;
            array<int_t, 2> &m_validity;

            template <class IterationPolicy,
                class Stages,
                bool IsFirst,
                bool IsLast,
                enable_if_t<meta::length<Stages>::value != 0, int> = 0>
            GT_FUNCTION void k_loop(int_t first, int_t last) const {
                for (int_t cur = first; IterationPolicy::condition(cur, last);
                     IterationPolicy::increment(cur), IterationPolicy::increment(m_domain)) {
                    if (m_in_domain)
                        m_domain.template fill_caches<IterationPolicy>(IsFirst && cur == first, m_validity);
                    RunEsfFunctor::template exec<Stages>(m_domain);
                    if (m_in_domain)
                        m_domain.template flush_caches<IterationPolicy>(IsLast && cur == last, m_validity);
                    m_domain.template slide_caches<IterationPolicy>();
                    IterationPolicy::decrement(m_validity[0]);
                    IterationPolicy::decrement(m_validity[1]);
                }
            }

            template <class IterationPolicy,
                class Stages,
                bool IsFirst,
                bool IsLast,
                enable_if_t<meta::length<Stages>::value == 0, int> = 0>
            GT_FUNCTION void k_loop(int_t first, int_t last) const {
                for (int_t cur = first; IterationPolicy::condition(cur, last);
                     IterationPolicy::increment(cur), IterationPolicy::increment(m_domain)) {
                    if (m_in_domain) {
                        m_domain.template fill_caches<IterationPolicy>(IsFirst && cur == first, m_validity);
                        m_domain.template flush_caches<IterationPolicy>(IsLast && cur == last, m_validity);
                    }
                    m_domain.template slide_caches<IterationPolicy>();
                    IterationPolicy::decrement(m_validity[0]);
                    IterationPolicy::decrement(m_validity[1]);
                }
            }

            template <class LoopInterval>
            GT_FUNCTION void operator()() const {
                GRIDTOOLS_STATIC_ASSERT(is_loop_interval<LoopInterval>::value, GT_INTERNAL_ERROR);
                using from_t = GT_META_CALL(meta::first, LoopInterval);
                using to_t = GT_META_CALL(meta::second, LoopInterval);
                using stage_groups_t = GT_META_CALL(meta::at_c, (LoopInterval, 2));
                using iteration_policy_t = iteration_policy<from_t, to_t, execution_type_t::iteration>;
                const auto k_interval = get_k_interval<from_t, to_t>(
                    typename RunFunctorArguments::backend_ids_t{}, execution_type_t{}, m_grid);
                constexpr auto is_first = std::is_same<LoopInterval, first_t>::value;
                constexpr auto is_last = std::is_same<LoopInterval, last_t>::value;
                k_loop<iteration_policy_t, stage_groups_t, is_first, is_last>(k_interval.first, k_interval.second);
            }
        };

        template <typename RunFunctorArguments, class RunEsfFunctor, class ItDomain, class Grid>
        struct run_f_on_interval {
            GRIDTOOLS_STATIC_ASSERT(is_run_functor_arguments<RunFunctorArguments>::value, GT_INTERNAL_ERROR);

            using execution_type_t = typename RunFunctorArguments::execution_type_t;

            ItDomain &m_domain;
            Grid const &m_grid;

            template <class IterationPolicy, class Stages, enable_if_t<meta::length<Stages>::value != 0, int> = 0>
            GT_FUNCTION void k_loop(int_t first, int_t last) const {
                for (int_t cur = first; IterationPolicy::condition(cur, last);
                     IterationPolicy::increment(cur), IterationPolicy::increment(m_domain))
                    RunEsfFunctor::template exec<Stages>(m_domain);
            }

            template <class IterationPolicy, class Stages, enable_if_t<meta::length<Stages>::value == 0, int> = 0>
            GT_FUNCTION void k_loop(int_t first, int_t last) const {
                // TODO(anstaf): supplement iteration_policy with the function that is functionally equivalent with
                //               this loop. smth. like: dry_run(from, to, it_domain);
                //
                // The weird thing here: because we use unnatural to C/C++ convention that the ranges are
                // defined not by [begin, end) but by [first, last], the implementation of this function would be
                // a bit messy [much more cryptic comparing to the current loop]. For me it is not clear what
                // to do first: fix an alien convention everywhere or implement this TODO.
                for (int_t cur = first; IterationPolicy::condition(cur, last);
                     IterationPolicy::increment(cur), IterationPolicy::increment(m_domain)) {
                }
            }

            template <class LoopInterval>
            GT_FUNCTION void operator()() const {
                GRIDTOOLS_STATIC_ASSERT(is_loop_interval<LoopInterval>::value, GT_INTERNAL_ERROR);
                using from_t = GT_META_CALL(meta::first, LoopInterval);
                using to_t = GT_META_CALL(meta::second, LoopInterval);
                using stage_groups_t = GT_META_CALL(meta::at_c, (LoopInterval, 2));
                using iteration_policy_t = iteration_policy<from_t, to_t, execution_type_t::iteration>;
                const auto k_interval = get_k_interval<from_t, to_t>(
                    typename RunFunctorArguments::backend_ids_t{}, execution_type_t{}, m_grid);
                k_loop<iteration_policy_t, stage_groups_t>(k_interval.first, k_interval.second);
            }
        };
    } // namespace _impl

    template <class RunFunctorArguments, class RunEsfFunctor, class ItDomain, class Grid>
    GT_FUNCTION enable_if_t<ItDomain::has_k_caches> run_functors_on_interval(ItDomain &it_domain, Grid const &grid) {
        using loop_intervals_t = typename RunFunctorArguments::loop_intervals_t;
        using first_t = GT_META_CALL(meta::first, loop_intervals_t);
        int_t first_pos = grid.template value_at<GT_META_CALL(meta::first, first_t)>();

        using last_t = GT_META_CALL(meta::last, loop_intervals_t);
        int_t last_pos = grid.template value_at<GT_META_CALL(meta::second, last_t)>();

        int_t steps = last_pos - first_pos;

        array<int_t, 2> validity{math::min(0, steps), math::max(0, steps)};

        bool in_domain = it_domain.template is_thread_in_domain<typename RunFunctorArguments::max_extent_t>();
        host_device::for_each_type<typename RunFunctorArguments::loop_intervals_t>(
            _impl::run_f_on_interval_with_k_caches<RunFunctorArguments, RunEsfFunctor, ItDomain, Grid>{
                it_domain, in_domain, grid, validity});
    }

    template <class RunFunctorArguments, class RunEsfFunctor, class ItDomain, class Grid>
    GT_FUNCTION enable_if_t<!ItDomain::has_k_caches> run_functors_on_interval(ItDomain &it_domain, Grid const &grid) {
        host_device::for_each_type<typename RunFunctorArguments::loop_intervals_t>(
            _impl::run_f_on_interval<RunFunctorArguments, RunEsfFunctor, ItDomain, Grid>{it_domain, grid});
    }

} // namespace gridtools

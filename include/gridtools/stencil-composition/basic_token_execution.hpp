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

            ItDomain &m_domain;
            Grid const &m_grid;

            template <class IterationPolicy, class Stages, enable_if_t<meta::length<Stages>::value != 0, int> = 0>
            GT_FUNCTION void k_loop(int_t first, int_t last, bool is_first, bool is_last) const {
                const bool in_domain =
                    m_domain.template is_thread_in_domain<typename RunFunctorArguments::max_extent_t>();

                for (int_t cur = first; IterationPolicy::condition(cur, last);
                     IterationPolicy::increment(cur), IterationPolicy::increment(m_domain)) {
                    if (in_domain)
                        m_domain.template fill_caches<IterationPolicy>(is_first && cur == first);

                    RunEsfFunctor::template exec<Stages>(m_domain);

                    if (in_domain) {
                        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 &&
                            blockIdx.z == 0 && threadIdx.z == 0)
                            printf("k_loop: cur = %i, last = %i\n", cur, last);
                        m_domain.template flush_caches<IterationPolicy>(is_last && cur == last);
                        m_domain.template slide_caches<IterationPolicy>();
                    }
                }
            }

            template <class IterationPolicy, class Stages, enable_if_t<meta::length<Stages>::value == 0, int> = 0>
            GT_FUNCTION void k_loop(int_t cur, int_t last, bool, bool) const {
                // TODO(anstaf): supplement iteration_policy with the function that is functionally equivalent with
                //               this loop. smth. like: dry_run(from, to, it_domain);
                //
                // The weird thing here: because we use unnatural to C/C++ convention that the ranges are
                // defined not by [begin, end) but by [first, last], the implementation of this function would be
                // a bit messy [much more cryptic comparing to the current loop]. For me it is not clear what
                // to do first: fix an alien convention everywhere or implement this TODO.
                for (; IterationPolicy::condition(cur, last);
                     IterationPolicy::increment(cur), IterationPolicy::increment(m_domain)) {
                }
            }

            template <class LoopInterval>
            GT_FUNCTION void operator()() const {
                GRIDTOOLS_STATIC_ASSERT(is_loop_interval<LoopInterval>::value, GT_INTERNAL_ERROR);
                using from_t = GT_META_CALL(meta::first, LoopInterval);
                using to_t = GT_META_CALL(meta::second, LoopInterval);
                using stage_groups_t = GT_META_CALL(meta::at_c, (LoopInterval, 2));
                using iteration_policy_t = iteration_policy<from_t, to_t, execution_engine::iteration>;
                using first_t = GT_META_CALL(meta::first, typename RunFunctorArguments::loop_intervals_t);
                using last_t = GT_META_CALL(meta::at_c,
                    (typename RunFunctorArguments::loop_intervals_t,
                        meta::length<typename RunFunctorArguments::loop_intervals_t>::value - 1));
                const auto k_interval = get_k_interval<from_t, to_t>(typename RunFunctorArguments::backend_ids_t{},
                    typename RunFunctorArguments::execution_type_t{},
                    m_grid);
                constexpr auto is_first = std::is_same<LoopInterval, first_t>::value;
                constexpr auto is_last = std::is_same<LoopInterval, last_t>::value;
                k_loop<iteration_policy_t, stage_groups_t>(k_interval.first, k_interval.second, is_first, is_last);
            }
        };
    } // namespace _impl

    template <class RunFunctorArguments, class RunEsfFunctor, class ItDomain, class Grid>
    GT_FUNCTION void run_functors_on_interval(ItDomain &it_domain, Grid const &grid) {
        host_device::for_each_type<typename RunFunctorArguments::loop_intervals_t>(
            _impl::run_f_on_interval<RunFunctorArguments, RunEsfFunctor, ItDomain, Grid>{it_domain, grid});
    }
} // namespace gridtools

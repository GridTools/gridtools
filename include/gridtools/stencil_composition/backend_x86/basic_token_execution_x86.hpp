/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

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

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../common/host_device.hpp"
#include "../../meta.hpp"
#include "../execution_types.hpp"
#include "../interval.hpp"
#include "../iteration_policy.hpp"
#include "../level.hpp"
#include "../loop_interval.hpp"
#include "../run_functor_arguments.hpp"

namespace gridtools {
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
        struct run_f_on_interval {
            GT_STATIC_ASSERT(is_run_functor_arguments<RunFunctorArguments>::value, GT_INTERNAL_ERROR);

            using execution_type_t = typename RunFunctorArguments::execution_type_t;

            ItDomain &m_domain;
            Grid const &m_grid;

            template <class IterationPolicy, class Stages, std::enable_if_t<meta::length<Stages>::value != 0, int> = 0>
            GT_FORCE_INLINE void k_loop(int_t first, int_t last) const {
                for (int_t cur = first; IterationPolicy::condition(cur, last);
                     IterationPolicy::increment(cur), IterationPolicy::increment(m_domain))
                    RunEsfFunctor::template exec<Stages>(m_domain);
            }

            template <class IterationPolicy, class Stages, std::enable_if_t<meta::length<Stages>::value == 0, int> = 0>
            GT_FORCE_INLINE void k_loop(int_t first, int_t last) const {
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
            GT_FORCE_INLINE void operator()() const {
                GT_STATIC_ASSERT(is_loop_interval<LoopInterval>::value, GT_INTERNAL_ERROR);
                using from_t = meta::first<LoopInterval>;
                using to_t = meta::second<LoopInterval>;
                using stage_groups_t = meta::at_c<LoopInterval, 2>;
                using iteration_policy_t = iteration_policy<from_t, to_t, execution_type_t>;
                k_loop<iteration_policy_t, stage_groups_t>(
                    m_grid.template value_at<from_t>(), m_grid.template value_at<to_t>());
            }
        };
    } // namespace _impl

    template <class RunFunctorArguments, class RunEsfFunctor, class ItDomain, class Grid>
    GT_FORCE_INLINE void run_functors_on_interval(ItDomain &it_domain, Grid const &grid) {
        for_each_type<typename RunFunctorArguments::loop_intervals_t>(
            _impl::run_f_on_interval<RunFunctorArguments, RunEsfFunctor, ItDomain, Grid>{it_domain, grid});
    }

} // namespace gridtools

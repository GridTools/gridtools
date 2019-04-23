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

#include "../../../meta.hpp"
//#include "../../backend_naive/basic_token_execution_naive.hpp"
#include "../../iteration_policy.hpp"
#include "../../pos3.hpp"
#include "../positional_iterate_domain.hpp"
#include "iterate_domain_naive.hpp"

/**@file
 * @brief mss loop implementations for the naive backend
 */
namespace gridtools {
    namespace mss_loop_naive_impl_ {
        template <class LocalDomain, class Grid>
        struct stage_executor_f {
            LocalDomain const &m_local_domain;
            Grid const &m_grid;

            template <template <class...> class L, class From, class To, class Stage>
            void operator()(L<From, To, Stage>) const {
                using extent_t = typename Stage::extent_t;
                int_t iminus = extent_t::iminus::value;
                int_t iplus = extent_t::iplus::value;
                int_t jminus = extent_t::jminus::value;
                int_t jplus = extent_t::jplus::value;
                int_t k_from = m_grid.template value_at<From>();
                int_t k_to = m_grid.template value_at<To>();
                int_t i_count = m_grid.i_high_bound() - m_grid.i_low_bound() + 1 + iplus - iminus;
                int_t j_count = m_grid.j_high_bound() - m_grid.j_low_bound() + 1 + jplus - jminus;
                int_t k_count = 1 + std::abs(k_to - k_from);
                int_t k_step = k_to >= k_from ? 1 : -1;

                iterate_domain_naive<LocalDomain> it_domain(m_local_domain, m_grid);

                it_domain.increment_i(iminus);
                it_domain.increment_j(jminus);
                it_domain.increment_k(k_from - m_grid.k_min());

                for (int_t i = 0; i != i_count; ++i) {
                    for (int_t j = 0; j != j_count; ++j) {
                        for (int_t k = 0; k != k_count; ++k) {
                            Stage::exec(it_domain);
                            it_domain.increment_k(k_step);
                        }
                        it_domain.increment_k(-k_count * k_step);
                        it_domain.increment_j();
                    }
                    it_domain.increment_j(-j_count);
                    it_domain.increment_i();
                }
            }
        };

        template <class LoopInterval,
            class From = GT_META_CALL(meta::first, LoopInterval),
            class To = GT_META_CALL(meta::second, LoopInterval),
            class StageGroups = GT_META_CALL(meta::third, LoopInterval),
            class Stages = GT_META_CALL(meta::flatten, StageGroups)>
        GT_META_DEFINE_ALIAS(
            split_loop_interval, meta::transform, (meta::curry<meta::list, From, To>::template apply, Stages));

        template <class LoopIntervals>
        GT_META_DEFINE_ALIAS(
            split_loop_intervals, meta::flatten, (GT_META_CALL(meta::transform, (split_loop_interval, LoopIntervals))));
    } // namespace mss_loop_naive_impl_

    /**
     * @brief main execution of a mss. Defines the IJ loop bounds of this particular block
     * and sequentially executes all the functors in the mss
     * @tparam RunFunctorArgs run functor arguments
     */
    template <class RunFunctorArgs, class LocalDomain, class Grid, class ExecutionInfo>
    void mss_loop(backend::naive, LocalDomain const &local_domain, Grid const &grid, ExecutionInfo) {
        GT_STATIC_ASSERT(is_run_functor_arguments<RunFunctorArgs>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(is_local_domain<LocalDomain>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);
        using loop_intervals_t =
            GT_META_CALL(mss_loop_naive_impl_::split_loop_intervals, typename RunFunctorArgs::loop_intervals_t);
        for_each<loop_intervals_t>(mss_loop_naive_impl_::stage_executor_f<LocalDomain, Grid>{local_domain, grid});
    }
} // namespace gridtools

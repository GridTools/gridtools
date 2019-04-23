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

#include "../../../common/generic_metafunctions/for_each.hpp"
#include "../../../meta.hpp"
#include "../../caches/cache_metafunctions.hpp"
#include "../../iteration_policy.hpp"
#include "../../loop_interval.hpp"
#include "./execinfo_mc.hpp"
#include "./iterate_domain_mc.hpp"

/**@file
 * @brief mss loop implementations for the mc backend
 */
namespace gridtools {
    namespace _impl_mss_loop_mc {
        /**
         * @brief Class for inner (block-level) looping.
         * Specialization for stencils with serial execution along k-axis and non-zero max extent.
         *
         * @tparam RunFunctorArgs Run functor arguments.
         * @tparam From K-axis level to start with.
         * @tparam To the last K-axis level to process.
         */
        template <typename ExecutionType, typename ItDomain, typename Grid, typename From, typename To>
        struct inner_functor_mc_kserial {
            ItDomain &m_it_domain;
            const Grid &m_grid;
            const execinfo_block_kserial_mc &m_execution_info;

            /**
             * @brief Executes the corresponding Stage
             */
            template <class Stage>
            GT_FORCE_INLINE void operator()(Stage) const {
                using iteration_policy_t = iteration_policy<From, To, ExecutionType>;
                using extent_t = typename Stage::extent_t;

                const int_t i_first = extent_t::iminus::value;
                const int_t i_last = m_execution_info.i_block_size + extent_t::iplus::value;
                const int_t j_first = extent_t::jminus::value;
                const int_t j_last = m_execution_info.j_block_size + extent_t::jplus::value;
                const int_t k_first = m_grid.template value_at<From>();
                const int_t k_last = m_grid.template value_at<To>();

                for (int_t j = j_first; j < j_last; ++j) {
                    m_it_domain.set_j_block_index(j);
                    for (int_t k = k_first; iteration_policy_t::condition(k, k_last);
                         iteration_policy_t::increment(k)) {
                        m_it_domain.set_k_block_index(k);
#ifdef NDEBUG
#pragma ivdep
#pragma omp simd
#endif
                        for (int_t i = i_first; i < i_last; ++i) {
                            m_it_domain.set_i_block_index(i);
                            Stage::exec(m_it_domain);
                        }
                    }
                }
            }
        };

        /**
         * @brief Class for inner (block-level) looping.
         * Specialization for stencils with parallel execution along k-axis.
         *
         * @tparam RunFunctorArgs Run functor arguments.
         * @tparam From K-axis level to start with.
         * @tparam To the last K-axis level to process.
         */
        template <typename ItDomain>
        struct inner_functor_mc_kparallel {
            ItDomain &m_it_domain;
            const execinfo_block_kparallel_mc &m_execution_info;

            /**
             * @brief Executes the corresponding functor on a single k-level inside the block.
             *
             * @param index Index in the functor list of the ESF functor that should be executed.
             */
            template <typename Stage>
            GT_FORCE_INLINE void operator()(Stage) const {
                using extent_t = typename Stage::extent_t;

                const int_t i_first = extent_t::iminus::value;
                const int_t i_last = m_execution_info.i_block_size + extent_t::iplus::value;
                const int_t j_first = extent_t::jminus::value;
                const int_t j_last = m_execution_info.j_block_size + extent_t::jplus::value;

                for (int_t j = j_first; j < j_last; ++j) {
                    m_it_domain.set_j_block_index(j);
#ifdef NDEBUG
#pragma ivdep
#pragma omp simd
#endif
                    for (int_t i = i_first; i < i_last; ++i) {
                        m_it_domain.set_i_block_index(i);
                        Stage::exec(m_it_domain);
                    }
                }
            }
        };

        /**
         * @brief Class for per-block looping on a single interval.
         */
        template <typename ExecutionType, typename ItDomain, typename Grid, typename ExecutionInfo>
        class interval_functor_mc;

        /**
         * @brief Class for per-block looping on a single interval.
         * Specialization for stencils with serial execution along k-axis and non-zero max extent.
         */
        template <typename ExecutionType, typename ItDomain, typename Grid>
        struct interval_functor_mc<ExecutionType, ItDomain, Grid, execinfo_block_kserial_mc> {
            ItDomain &m_it_domain;
            Grid const &m_grid;
            execinfo_block_kserial_mc const &m_execution_info;

            template <class From, class To, class StageGroups>
            GT_FORCE_INLINE void operator()(loop_interval<From, To, StageGroups>) const {
                gridtools::for_each<GT_META_CALL(meta::flatten, StageGroups)>(
                    inner_functor_mc_kserial<ExecutionType, ItDomain, Grid, From, To>{
                        m_it_domain, m_grid, m_execution_info});
            }
        };

        /**
         * @brief Class for per-block looping on a single interval.
         * Specialization for stencils with parallel execution along k-axis.
         */
        template <typename ExecutionType, typename ItDomain, typename Grid>
        class interval_functor_mc<ExecutionType, ItDomain, Grid, execinfo_block_kparallel_mc> {
            ItDomain &m_it_domain;
            Grid const &m_grid;
            const execinfo_block_kparallel_mc &m_execution_info;

          public:
            GT_FORCE_INLINE interval_functor_mc(
                ItDomain &it_domain, Grid const &grid, execinfo_block_kparallel_mc const &execution_info)
                : m_it_domain(it_domain), m_grid(grid), m_execution_info(execution_info) {
                m_it_domain.set_k_block_index(m_execution_info.k);
            }

            template <class From, class To, class StageGroups>
            GT_FORCE_INLINE void operator()(loop_interval<From, To, StageGroups>) const {
                const int_t k_first = this->m_grid.template value_at<From>();
                const int_t k_last = this->m_grid.template value_at<To>();

                if (k_first <= m_execution_info.k && m_execution_info.k <= k_last)
                    gridtools::for_each<GT_META_CALL(meta::flatten, StageGroups)>(
                        inner_functor_mc_kparallel<ItDomain>{m_it_domain, m_execution_info});
            }
        };

    } // namespace _impl_mss_loop_mc

    /**
     * @brief main execution of a mss. Defines the IJ loop bounds of this particular block
     * and sequentially executes all the functors in the mss
     * @tparam RunFunctorArgs run functor arguments
     */
    template <class RunFunctorArgs, class LocalDomain, class Grid, class ExecutionInfo>
    GT_FORCE_INLINE static void mss_loop(
        backend::mc const &, LocalDomain const &local_domain, Grid const &grid, const ExecutionInfo &execution_info) {
        GT_STATIC_ASSERT(is_run_functor_arguments<RunFunctorArgs>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(is_local_domain<LocalDomain>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);
        using ij_cached_args_t = conditional_t<std::is_same<ExecutionInfo, execinfo_block_kparallel_mc>::value,
            GT_META_CALL(ij_cache_args, typename LocalDomain::cache_sequence_t),
            meta::list<>>;

        using iterate_domain_t = iterate_domain_mc<LocalDomain, ij_cached_args_t>;

        iterate_domain_t it_domain(local_domain, execution_info.i_first, execution_info.j_first);

        host::for_each<typename RunFunctorArgs::loop_intervals_t>(_impl_mss_loop_mc::
                interval_functor_mc<typename RunFunctorArgs::execution_type_t, iterate_domain_t, Grid, ExecutionInfo>{
                    it_domain, grid, execution_info});
    }
} // namespace gridtools

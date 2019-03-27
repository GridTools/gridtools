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
#include "../../iteration_policy.hpp"
#include "../../loop_interval.hpp"
#include "./execinfo_mc.hpp"
#include "./iterate_domain_mc.hpp"

/**@file
 * @brief mss loop implementations for the mc backend
 */
namespace gridtools {
    namespace _impl_mss_loop_mc {
        template <class RunFunctorArgs>
        GT_META_DEFINE_ALIAS(get_iterate_domain_type,
            meta::id,
            (iterate_domain_mc<iterate_domain_arguments<target::mc,
                    typename RunFunctorArgs::local_domain_t,
                    typename RunFunctorArgs::esf_sequence_t,
                    typename RunFunctorArgs::cache_sequence_t,
                    typename RunFunctorArgs::grid_t>>));

        /**
         * @brief Meta function to check if all ESFs can be computed independently per column. This is possible if the
         * max extent is zero, i.e. there are no dependencies between the ESFs with offsets along i or j.
         */
        template <typename RunFunctorArgs>
        using enable_inner_k_fusion = std::integral_constant<bool,
            RunFunctorArgs::max_extent_t::iminus::value == 0 && RunFunctorArgs::max_extent_t::iplus::value == 0 &&
                RunFunctorArgs::max_extent_t::jminus::value == 0 && RunFunctorArgs::max_extent_t::jplus::value == 0 &&
                /* TODO: enable. This is currently disabled as the performance is not satisfying, probably due to
                   missing vector-sized k-caches and possibly alignment issues. */
                false>;

        constexpr int_t veclength_mc = 16;

        /**
         * @brief Class for inner (block-level) looping.
         * Specialization for stencils with serial execution along k-axis and max extent = 0.
         *
         * @tparam RunFunctorArgs Run functor arguments.
         * @tparam From K-axis level to start with.
         * @tparam To the last K-axis level to process.
         */
        template <typename RunFunctorArgs, typename From, typename To>
        class inner_functor_mc_kserial_fused {
            using grid_t = typename RunFunctorArgs::grid_t;
            using iterate_domain_t = typename RunFunctorArgs::iterate_domain_t;

          public:
            GT_FORCE_INLINE inner_functor_mc_kserial_fused(iterate_domain_t &it_domain,
                const grid_t &grid,
                const execinfo_block_kserial_mc &execution_info,
                int_t i_vecfirst,
                int_t i_veclast)
                : m_it_domain(it_domain), m_grid(grid), m_execution_info(execution_info), m_i_vecfirst(i_vecfirst),
                  m_i_veclast(i_veclast) {}

            /**
             * @brief Executes the corresponding functor in the given interval.
             *
             * @param index Index in the functor list of the ESF functor that should be executed.
             */
            template <template <class...> class L, class Stage, class Index>
            GT_FORCE_INLINE void operator()(L<Stage, Index>) const {
                using execution_type_t = typename RunFunctorArgs::execution_type_t;
                using iteration_policy_t = iteration_policy<From, To, execution_type_t>;

                const int_t k_first = m_grid.template value_at<From>();
                const int_t k_last = m_grid.template value_at<To>();

                /* Prefetching is only done for the first ESF, as we assume the following ESFs access the same data
                 * that's already in cache then. The prefetching distance is currently always 2 k-levels. */
                if (Index::value == 0)
                    m_it_domain.set_prefetch_distance(k_first <= k_last ? 2 : -2);

                for (int_t k = k_first; iteration_policy_t::condition(k, k_last); iteration_policy_t::increment(k)) {
                    m_it_domain.set_k_block_index(k);
#ifdef NDEBUG
#pragma ivdep
#pragma omp simd
#endif
                    for (int_t i = m_i_vecfirst; i < m_i_veclast; ++i) {
                        m_it_domain.set_i_block_index(i);
                        Stage::exec(m_it_domain);
                    }
                }

                m_it_domain.set_prefetch_distance(0);
            }

          private:
            iterate_domain_t &m_it_domain;
            const grid_t &m_grid;
            const execinfo_block_kserial_mc &m_execution_info;
            const int_t m_i_vecfirst, m_i_veclast;
        };

        /**
         * @brief Class for inner (block-level) looping.
         * Specialization for stencils with serial execution along k-axis and non-zero max extent.
         *
         * @tparam RunFunctorArgs Run functor arguments.
         * @tparam From K-axis level to start with.
         * @tparam To the last K-axis level to process.
         */
        template <typename RunFunctorArgs, typename From, typename To>
        class inner_functor_mc_kserial {
            using grid_t = typename RunFunctorArgs::grid_t;
            using iterate_domain_t = GT_META_CALL(get_iterate_domain_type, RunFunctorArgs);

          public:
            GT_FORCE_INLINE inner_functor_mc_kserial(
                iterate_domain_t &it_domain, const grid_t &grid, const execinfo_block_kserial_mc &execution_info)
                : m_it_domain(it_domain), m_grid(grid), m_execution_info(execution_info) {}

            /**
             * @brief Executes the corresponding Stage
             */
            template <class Stage>
            GT_FORCE_INLINE void operator()(Stage) const {
                using execution_type_t = typename RunFunctorArgs::execution_type_t;
                using iteration_policy_t = iteration_policy<From, To, execution_type_t>;
                using extent_t = typename Stage::extent_t;

                const int_t i_first = extent_t::iminus::value;
                const int_t i_last = m_execution_info.i_block_size + extent_t::iplus::value;
                const int_t j_first = extent_t::jminus::value;
                const int_t j_last = m_execution_info.j_block_size + extent_t::jplus::value;
                const int_t k_first = m_grid.template value_at<From>();
                const int_t k_last = m_grid.template value_at<To>();

                m_it_domain.set_block_base(m_execution_info.i_first, m_execution_info.j_first);
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

          private:
            iterate_domain_t &m_it_domain;
            const grid_t &m_grid;
            const execinfo_block_kserial_mc &m_execution_info;
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

                m_it_domain.set_block_base(m_execution_info.i_first, m_execution_info.j_first);
                m_it_domain.set_k_block_index(m_execution_info.k);
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
        template <typename RunFunctorArgs, typename ExecutionInfo, typename Enable = void>
        class interval_functor_mc;

        /**
         * @brief Class for per-block looping on a single interval.
         * Specialization for stencils with serial execution along k-axis and max extent of 0.
         */
        template <typename RunFunctorArgs>
        class interval_functor_mc<RunFunctorArgs,
            execinfo_block_kserial_mc,
            enable_if_t<enable_inner_k_fusion<RunFunctorArgs>::value>> {
            using grid_t = typename RunFunctorArgs::grid_t;
            using iterate_domain_t = GT_META_CALL(get_iterate_domain_type, RunFunctorArgs);

          public:
            GT_FORCE_INLINE interval_functor_mc(
                iterate_domain_t &it_domain, const grid_t &grid, const execinfo_block_kserial_mc &execution_info)
                : m_it_domain(it_domain), m_grid(grid), m_execution_info(execution_info) {}

            /**
             * @brief Runs all functors in RunFunctorArgs on the given interval.
             */
            template <class From, class To, class StageGroups>
            GT_FORCE_INLINE void operator()(loop_interval<From, To, StageGroups>) const {
                using extent_t = typename RunFunctorArgs::max_extent_t;
                using stages_t = GT_META_CALL(meta::flatten, StageGroups);
                using indices_t = GT_META_CALL(meta::make_indices_for, stages_t);
                using stages_and_indices_t = GT_META_CALL(meta::zip, (stages_t, indices_t));
                using inner_functor_t = inner_functor_mc_kserial_fused<RunFunctorArgs, From, To>;

                const int_t i_first = extent_t::iminus::value;
                const int_t i_last = m_execution_info.i_block_size + extent_t::iplus::value;
                const int_t j_first = extent_t::jminus::value;
                const int_t j_last = m_execution_info.j_block_size + extent_t::jplus::value;

                m_it_domain.set_block_base(m_execution_info.i_first, m_execution_info.j_first);
                for (int_t j = j_first; j < j_last; ++j) {
                    m_it_domain.set_j_block_index(j);
                    for (int_t i_vecfirst = i_first; i_vecfirst < i_last; i_vecfirst += veclength_mc) {
                        const int_t i_veclast = i_vecfirst + veclength_mc > i_last ? i_last : i_vecfirst + veclength_mc;
                        gridtools::for_each<stages_and_indices_t>(
                            inner_functor_t(m_it_domain, m_grid, m_execution_info, i_vecfirst, i_veclast));
                    }
                }
            }

          private:
            iterate_domain_t &m_it_domain;
            const grid_t &m_grid;
            const execinfo_block_kserial_mc &m_execution_info;
        };

        /**
         * @brief Class for per-block looping on a single interval.
         * Specialization for stencils with serial execution along k-axis and non-zero max extent.
         */
        template <typename RunFunctorArgs>
        class interval_functor_mc<RunFunctorArgs,
            execinfo_block_kserial_mc,
            enable_if_t<!enable_inner_k_fusion<RunFunctorArgs>::value>> {
            using grid_t = typename RunFunctorArgs::grid_t;
            using iterate_domain_t = GT_META_CALL(get_iterate_domain_type, RunFunctorArgs);

          public:
            GT_FORCE_INLINE interval_functor_mc(
                iterate_domain_t &it_domain, const grid_t &grid, const execinfo_block_kserial_mc &execution_info)
                : m_it_domain(it_domain), m_grid(grid), m_execution_info(execution_info) {}

            /**
             * @brief Runs all functors in RunFunctorArgs on the given interval.
             */
            template <class From, class To, class StageGroups>
            GT_FORCE_INLINE void operator()(loop_interval<From, To, StageGroups>) const {
                gridtools::for_each<GT_META_CALL(meta::flatten, StageGroups)>(
                    inner_functor_mc_kserial<RunFunctorArgs, From, To>(m_it_domain, m_grid, m_execution_info));
            }

          private:
            iterate_domain_t &m_it_domain;
            const grid_t &m_grid;
            const execinfo_block_kserial_mc &m_execution_info;
        };

        /**
         * @brief Class for per-block looping on a single interval.
         * Specialization for stencils with parallel execution along k-axis.
         */
        template <typename RunFunctorArgs, typename Enable>
        class interval_functor_mc<RunFunctorArgs, execinfo_block_kparallel_mc, Enable> {
            using grid_t = typename RunFunctorArgs::grid_t;
            using iterate_domain_t = GT_META_CALL(get_iterate_domain_type, RunFunctorArgs);

          public:
            GT_FORCE_INLINE interval_functor_mc(
                iterate_domain_t &it_domain, const grid_t &grid, const execinfo_block_kparallel_mc &execution_info)
                : m_it_domain(it_domain), m_grid(grid), m_execution_info(execution_info) {
                // enable ij-caches
                m_it_domain.enable_ij_caches();
            }

            /**
             * @brief Runs all functors in RunFunctorArgs on the given interval if k is inside the interval.
             */
            template <class From, class To, class StageGroups>
            GT_FORCE_INLINE void operator()(loop_interval<From, To, StageGroups>) const {
                const int_t k_first = this->m_grid.template value_at<From>();
                const int_t k_last = this->m_grid.template value_at<To>();

                if (k_first <= m_execution_info.k && m_execution_info.k <= k_last)
                    gridtools::for_each<GT_META_CALL(meta::flatten, StageGroups)>(
                        inner_functor_mc_kparallel<iterate_domain_t>{m_it_domain, m_execution_info});
            }

          private:
            iterate_domain_t &m_it_domain;
            const grid_t &m_grid;
            const execinfo_block_kparallel_mc &m_execution_info;
        };

    } // namespace _impl_mss_loop_mc

    /**
     * @brief main execution of a mss. Defines the IJ loop bounds of this particular block
     * and sequentially executes all the functors in the mss
     * @tparam RunFunctorArgs run functor arguments
     */
    template <class RunFunctorArgs, class LocalDomain, class Grid, class ExecutionInfo>
    GT_FORCE_INLINE static void mss_loop(
        target::mc const &, LocalDomain const &local_domain, Grid const &grid, const ExecutionInfo &execution_info) {
        GT_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((is_local_domain<LocalDomain>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);
        using iterate_domain_t = GT_META_CALL(_impl_mss_loop_mc::get_iterate_domain_type, RunFunctorArgs);

        iterate_domain_t it_domain(local_domain);

        host::for_each<typename RunFunctorArgs::loop_intervals_t>(
            _impl_mss_loop_mc::interval_functor_mc<RunFunctorArgs, ExecutionInfo>(it_domain, grid, execution_info));
    }
} // namespace gridtools

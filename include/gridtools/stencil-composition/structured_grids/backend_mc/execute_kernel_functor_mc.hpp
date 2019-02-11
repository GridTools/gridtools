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

#include "../../../common/generic_metafunctions/for_each.hpp"
#include "../../../meta.hpp"
#include "../../grid_traits.hpp"
#include "../../iteration_policy.hpp"
#include "../../loop_interval.hpp"
#include "./execinfo_mc.hpp"
#include "./iterate_domain_mc.hpp"

namespace gridtools {

    namespace _impl {

        template <class RunFunctorArguments>
        GT_META_DEFINE_ALIAS(get_iterate_domain_type,
            meta::id,
            (iterate_domain_mc<iterate_domain_arguments<typename RunFunctorArguments::backend_ids_t,
                    typename RunFunctorArguments::local_domain_t,
                    typename RunFunctorArguments::esf_sequence_t,
                    typename RunFunctorArguments::extent_sizes_t,
                    typename RunFunctorArguments::max_extent_t,
                    typename RunFunctorArguments::cache_sequence_t,
                    typename RunFunctorArguments::grid_t>>));

        /**
         * @brief Meta function to check if all ESFs can be computed independently per column. This is possible if the
         * max extent is zero, i.e. there are no dependencies between the ESFs with offsets along i or j.
         */
        template <typename RunFunctorArguments>
        using enable_inner_k_fusion = std::integral_constant<bool,
            RunFunctorArguments::max_extent_t::iminus::value == 0 &&
                RunFunctorArguments::max_extent_t::iplus::value == 0 &&
                RunFunctorArguments::max_extent_t::jminus::value == 0 &&
                RunFunctorArguments::max_extent_t::jplus::value == 0 &&
                /* TODO: enable. This is currently disabled as the performance is not satisfying, probably due to
                   missing vector-sized k-caches and possibly alignment issues. */
                false>;

        constexpr int_t veclength_mc = 16;

        /**
         * @brief Class for inner (block-level) looping.
         * Specialization for stencils with serial execution along k-axis and max extent = 0.
         *
         * @tparam RunFunctorArguments Run functor arguments.
         * @tparam From K-axis level to start with.
         * @tparam To the last K-axis level to process.
         */
        template <typename RunFunctorArguments, typename From, typename To>
        class inner_functor_mc_kserial_fused {
            using grid_t = typename RunFunctorArguments::grid_t;
            using iterate_domain_t = typename RunFunctorArguments::iterate_domain_t;

          public:
            GT_FUNCTION inner_functor_mc_kserial_fused(iterate_domain_t &it_domain,
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
            GT_FUNCTION void operator()(L<Stage, Index>) const {
                typename RunFunctorArguments::execution_type_t::bla tmp;
                using execution_type_t = typename RunFunctorArguments::execution_type_t;
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
         * @tparam RunFunctorArguments Run functor arguments.
         * @tparam From K-axis level to start with.
         * @tparam To the last K-axis level to process.
         */
        template <typename RunFunctorArguments, typename From, typename To>
        class inner_functor_mc_kserial {
            using grid_t = typename RunFunctorArguments::grid_t;
            using iterate_domain_t = GT_META_CALL(get_iterate_domain_type, RunFunctorArguments);

          public:
            GT_FUNCTION inner_functor_mc_kserial(
                iterate_domain_t &it_domain, const grid_t &grid, const execinfo_block_kserial_mc &execution_info)
                : m_it_domain(it_domain), m_grid(grid), m_execution_info(execution_info) {}

            /**
             * @brief Executes the corresponding Stage
             */
            template <class Stage>
            GT_FUNCTION void operator()(Stage) const {
                using execution_type_t = typename RunFunctorArguments::execution_type_t;
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
         * @tparam RunFunctorArguments Run functor arguments.
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
            GT_FUNCTION void operator()(Stage) const {
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
        template <typename RunFunctorArguments, typename ExecutionInfo, typename Enable = void>
        class interval_functor_mc;

        /**
         * @brief Class for per-block looping on a single interval.
         * Specialization for stencils with serial execution along k-axis and max extent of 0.
         */
        template <typename RunFunctorArguments>
        class interval_functor_mc<RunFunctorArguments,
            execinfo_block_kserial_mc,
            enable_if_t<enable_inner_k_fusion<RunFunctorArguments>::value>> {
            using grid_t = typename RunFunctorArguments::grid_t;
            using iterate_domain_t = GT_META_CALL(get_iterate_domain_type, RunFunctorArguments);

          public:
            GT_FUNCTION interval_functor_mc(
                iterate_domain_t &it_domain, const grid_t &grid, const execinfo_block_kserial_mc &execution_info)
                : m_it_domain(it_domain), m_grid(grid), m_execution_info(execution_info) {}

            /**
             * @brief Runs all functors in RunFunctorArguments on the given interval.
             */
            template <class From, class To, class StageGroups>
            GT_FUNCTION void operator()(loop_interval<From, To, StageGroups>) const {
                using extent_t = typename RunFunctorArguments::max_extent_t;
                using stages_t = GT_META_CALL(meta::flatten, StageGroups);
                using indices_t = GT_META_CALL(meta::make_indices_for, stages_t);
                using stages_and_indices_t = GT_META_CALL(meta::zip, (stages_t, indices_t));
                using inner_functor_t = inner_functor_mc_kserial_fused<RunFunctorArguments, From, To>;

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
        template <typename RunFunctorArguments>
        class interval_functor_mc<RunFunctorArguments,
            execinfo_block_kserial_mc,
            enable_if_t<!enable_inner_k_fusion<RunFunctorArguments>::value>> {
            using grid_t = typename RunFunctorArguments::grid_t;
            using iterate_domain_t = GT_META_CALL(get_iterate_domain_type, RunFunctorArguments);

          public:
            GT_FUNCTION interval_functor_mc(
                iterate_domain_t &it_domain, const grid_t &grid, const execinfo_block_kserial_mc &execution_info)
                : m_it_domain(it_domain), m_grid(grid), m_execution_info(execution_info) {}

            /**
             * @brief Runs all functors in RunFunctorArguments on the given interval.
             */
            template <class From, class To, class StageGroups>
            GT_FUNCTION void operator()(loop_interval<From, To, StageGroups>) const {
                gridtools::for_each<GT_META_CALL(meta::flatten, StageGroups)>(
                    inner_functor_mc_kserial<RunFunctorArguments, From, To>(m_it_domain, m_grid, m_execution_info));
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
        template <typename RunFunctorArguments, typename Enable>
        class interval_functor_mc<RunFunctorArguments, execinfo_block_kparallel_mc, Enable> {
            using grid_t = typename RunFunctorArguments::grid_t;
            using iterate_domain_t = GT_META_CALL(get_iterate_domain_type, RunFunctorArguments);

          public:
            GT_FUNCTION interval_functor_mc(
                iterate_domain_t &it_domain, const grid_t &grid, const execinfo_block_kparallel_mc &execution_info)
                : m_it_domain(it_domain), m_grid(grid), m_execution_info(execution_info) {
                // enable ij-caches
                m_it_domain.enable_ij_caches();
            }

            /**
             * @brief Runs all functors in RunFunctorArguments on the given interval if k is inside the interval.
             */
            template <class From, class To, class StageGroups>
            GT_FUNCTION void operator()(loop_interval<From, To, StageGroups>) const {
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

    } // namespace _impl

    namespace strgrid {

        /**
         * @brief Class for executing all functors on a single block.
         */
        template <typename RunFunctorArguments>
        class execute_kernel_functor_mc {
            using grid_t = typename RunFunctorArguments::grid_t;
            using local_domain_t = typename RunFunctorArguments::local_domain_t;

          public:
            GT_FUNCTION execute_kernel_functor_mc(const local_domain_t &local_domain, const grid_t &grid)
                : m_local_domain(local_domain), m_grid(grid) {}

            template <class ExecutionInfo>
            GT_FUNCTION void operator()(const ExecutionInfo &execution_info) const {
                using iterate_domain_t = GT_META_CALL(gridtools::_impl::get_iterate_domain_type, RunFunctorArguments);

                iterate_domain_t it_domain(m_local_domain);

                gridtools::for_each<typename RunFunctorArguments::loop_intervals_t>(
                    gridtools::_impl::interval_functor_mc<RunFunctorArguments, ExecutionInfo>(
                        it_domain, m_grid, execution_info));
            }

          private:
            const local_domain_t &m_local_domain;
            const grid_t &m_grid;
        };

    } // namespace strgrid
} // namespace gridtools

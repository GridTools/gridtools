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

#include <type_traits>

#include "../../../common/generic_metafunctions/for_each.hpp"
#include "../../../meta.hpp"
#include "../../caches/cache_metafunctions.hpp"
#include "../../loop_interval.hpp"
#include "../../run_functor_arguments.hpp"
#include "execinfo_mc.hpp"
#include "iterate_domain_mc.hpp"

/**@file
 * @brief mss loop implementations for the mc backend
 */
namespace gridtools {
    namespace _impl_mss_loop_mc {
        using namespace literals;

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
            GT_FORCE_INLINE void operator()(Stage stage) const {
                using extent_t = typename Stage::extent_t;
                auto j_count = m_execution_info.j_block_size + extent_t::jplus::value - extent_t::jminus::value;
                auto i_count = m_execution_info.i_block_size + extent_t::iplus::value - extent_t::iminus::value;
                int_t k_first = m_grid.template value_at<From>();
                int_t k_last = m_grid.template value_at<To>();
                auto k_count = 1 + std::abs(k_last - k_first);
                static constexpr std::conditional_t<execute::is_backward<ExecutionType>::value,
                    integral_constant<int_t, -1>,
                    integral_constant<int_t, 1>>
                    k_step = {};

                auto ptr = m_it_domain.ptr();
                const auto &strides = m_it_domain.strides();

                sid::shift(ptr, sid::get_stride<dim::i>(strides), integral_constant<int_t, extent_t::iminus::value>{});
                sid::shift(ptr, sid::get_stride<dim::j>(strides), integral_constant<int_t, extent_t::jminus::value>{});
                sid::shift(ptr, sid::get_stride<dim::k>(strides), k_first);

                for (int_t j = 0; j < j_count; ++j) {
                    for (int_t k = 0; k < k_count; ++k) {
#ifdef NDEBUG
#pragma ivdep
#ifndef __INTEL_COMPILER
#pragma omp simd
#endif
#endif
                        for (int_t i = 0; i < i_count; ++i) {
                            stage(ptr, strides);
                            sid::shift(ptr, sid::get_stride<dim::i>(strides), 1_c);
                        }
                        sid::shift(ptr, sid::get_stride<dim::i>(strides), -i_count);
                        sid::shift(ptr, sid::get_stride<dim::k>(strides), k_step);
                    }
                    sid::shift(ptr, sid::get_stride<dim::k>(strides), -k_count * k_step);
                    sid::shift(ptr, sid::get_stride<dim::j>(strides), 1_c);
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
            GT_FORCE_INLINE void operator()(Stage stage) const {
                using extent_t = typename Stage::extent_t;
                auto j_count = m_execution_info.j_block_size + extent_t::jplus::value - extent_t::jminus::value;
                auto i_count = m_execution_info.i_block_size + extent_t::iplus::value - extent_t::iminus::value;

                auto ptr = m_it_domain.ptr();
                const auto &strides = m_it_domain.strides();

                sid::shift(ptr, sid::get_stride<dim::i>(strides), integral_constant<int_t, extent_t::iminus::value>{});
                sid::shift(ptr, sid::get_stride<dim::j>(strides), integral_constant<int_t, extent_t::jminus::value>{});
                m_it_domain.k_shift(ptr, m_execution_info.k);

                for (int_t j = 0; j < j_count; ++j) {
#ifdef NDEBUG
#pragma ivdep
#ifndef __INTEL_COMPILER
#pragma omp simd
#endif
#endif
                    for (int_t i = 0; i < i_count; ++i) {
                        stage(ptr, strides);
                        sid::shift(ptr, sid::get_stride<dim::i>(strides), 1_c);
                    }
                    sid::shift(ptr, sid::get_stride<dim::i>(strides), -i_count);
                    sid::shift(ptr, sid::get_stride<dim::j>(strides), 1_c);
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

            template <class From, class To, class Stages>
            GT_FORCE_INLINE void operator()(loop_interval<From, To, Stages>) const {
                gridtools::for_each<Stages>(inner_functor_mc_kserial<ExecutionType, ItDomain, Grid, From, To>{
                    m_it_domain, m_grid, m_execution_info});
            }
        };

        /**
         * @brief Class for per-block looping on a single interval.
         * Specialization for stencils with parallel execution along k-axis.
         */
        template <typename ExecutionType, typename ItDomain, typename Grid>
        struct interval_functor_mc<ExecutionType, ItDomain, Grid, execinfo_block_kparallel_mc> {
            ItDomain &m_it_domain;
            Grid const &m_grid;
            const execinfo_block_kparallel_mc &m_execution_info;

            template <class From, class To, class Stages>
            GT_FORCE_INLINE void operator()(loop_interval<From, To, Stages>) const {
                if (m_execution_info.k < m_grid.template value_at<From>() ||
                    m_execution_info.k > m_grid.template value_at<To>())
                    return;
                gridtools::for_each<Stages>(inner_functor_mc_kparallel<ItDomain>{m_it_domain, m_execution_info});
            }
        };

    } // namespace _impl_mss_loop_mc

    /**
     * @brief main execution of a mss. Defines the IJ loop bounds of this particular block
     * and sequentially executes all the functors in the mss
     * @tparam RunFunctorArgs run functor arguments
     */
    template <class RunFunctorArgs, class LocalDomain, class Grid, class ExecutionInfo>
    GT_FORCE_INLINE void mss_loop(
        backend::mc, LocalDomain const &local_domain, Grid const &grid, const ExecutionInfo &execution_info) {
        GT_STATIC_ASSERT(is_run_functor_arguments<RunFunctorArgs>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(is_local_domain<LocalDomain>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);
        using ij_cached_args_t = std::conditional_t<std::is_same<ExecutionInfo, execinfo_block_kparallel_mc>::value,
            ij_cache_args<typename LocalDomain::cache_sequence_t>,
            meta::list<>>;

        using iterate_domain_t = iterate_domain_mc<LocalDomain, ij_cached_args_t>;

        iterate_domain_t it_domain(local_domain, execution_info.i_first, execution_info.j_first);

        host::for_each<typename RunFunctorArgs::loop_intervals_t>(_impl_mss_loop_mc::
                interval_functor_mc<typename RunFunctorArgs::execution_type_t, iterate_domain_t, Grid, ExecutionInfo>{
                    it_domain, grid, execution_info});
    }
} // namespace gridtools

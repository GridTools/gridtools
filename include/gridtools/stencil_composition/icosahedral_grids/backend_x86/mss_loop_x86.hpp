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

#include "../../../common/defs.hpp"
#include "../../../common/host_device.hpp"
#include "../../../meta.hpp"
#include "../../backend_x86/basic_token_execution_x86.hpp"
#include "../../loop_interval.hpp"
#include "../../pos3.hpp"
#include "../esf_metafunctions.hpp"
#include "../stage.hpp"
#include "iterate_domain_x86.hpp"

/**@file
 * @brief mss loop implementations for the x86 backend
 */
namespace gridtools {
    namespace _impl_mss_loop_x86 {
        template <size_t Color>
        struct loop_interval_contains_color {
            template <class T>
            using apply = meta::any_of<stage_group_contains_color<Color>::template apply, meta::at_c<T, 2>>;
        };

        template <uint_t Color>
        struct run_esf_functor_x86 {
            template <class StageGroups, class ItDomain>
            GT_FORCE_INLINE static void exec(ItDomain &it_domain) {
                using stages_t = meta::flatten<StageGroups>;
                GT_STATIC_ASSERT(meta::length<stages_t>::value == 1, GT_INTERNAL_ERROR);
                using stage_t = meta::first<stages_t>;
                stage_t::template exec<Color>(it_domain);
            }
        };

        template <class LoopIntervals>
        struct get_ncolors;

        // In the x86 case loop intervals contains a single stage.
        template <template <class...> class L0,
            template <class...> class L1,
            template <class...> class L2,
            class From,
            class To,
            class Stage>
        struct get_ncolors<L0<loop_interval<From, To, L1<L2<Stage>>>>> : Stage::n_colors {};

        /**
         * @tparam RunFunctorArgs run functor argument type with the main configuration of the MSS
         * @tparam IterateDomain iterator domain class
         * @tparam Grid grid object as it provided by user.
         */
        template <typename RunFunctorArgs, typename IterateDomain, typename Grid>
        struct color_execution_functor {
            GT_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value), GT_INTERNAL_ERROR);
            GT_STATIC_ASSERT((is_iterate_domain<IterateDomain>::value), GT_INTERNAL_ERROR);
            GT_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);

            template <class Color>
            using has_color = meta::any_of<loop_interval_contains_color<Color::value>::template apply,
                typename RunFunctorArgs::loop_intervals_t>;

            IterateDomain &m_it_domain;
            Grid const &m_grid;
            int_t m_loop_size;

            template <class Color, std::enable_if_t<has_color<Color>::value, int> = 0>
            void operator()(Color) const {
                for (int_t j = 0; j != m_loop_size; ++j) {
                    auto memorized_index = m_it_domain.index();
                    run_functors_on_interval<RunFunctorArgs, run_esf_functor_x86<Color::value>>(m_it_domain, m_grid);
                    m_it_domain.set_index(memorized_index);
                    m_it_domain.increment_j();
                }
                m_it_domain.increment_j(-m_loop_size);
                m_it_domain.increment_c();
            }
            template <class Color, std::enable_if_t<!has_color<Color>::value, int> = 0>
            void operator()(Color) const {
                // If there is no ESF in the sequence matching the color, we skip execution and simply increment the
                // color iterator
                m_it_domain.increment_c();
            }
        };

    } // namespace _impl_mss_loop_x86

    /**
     * @brief main execution of a mss. Defines the IJ loop bounds of this particular block
     * and sequentially executes all the functors in the mss
     * @tparam RunFunctorArgs run functor arguments
     */
    template <class RunFunctorArgs, class LocalDomain, class Grid, class ExecutionInfo>
    GT_FORCE_INLINE static void mss_loop(backend::x86 const &backend_target,
        LocalDomain const &local_domain,
        Grid const &grid,
        const ExecutionInfo &execution_info) {
        GT_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((is_local_domain<LocalDomain>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);

        using iterate_domain_arguments_t =
            iterate_domain_arguments<backend::x86, LocalDomain, typename RunFunctorArgs::esf_sequence_t>;
        using iterate_domain_t = iterate_domain_x86<iterate_domain_arguments_t>;
        iterate_domain_t it_domain(local_domain);

        using extent_t = get_extent_from_loop_intervals<typename RunFunctorArgs::loop_intervals_t>;
        using interval_t = meta::first<typename RunFunctorArgs::loop_intervals_t>;
        using from_t = meta::first<interval_t>;
        it_domain.initialize({grid.i_low_bound(), grid.j_low_bound(), grid.k_min()},
            {execution_info.bi, execution_info.bj, 0},
            {extent_t::iminus::value, extent_t::jminus::value, grid.template value_at<from_t>() - grid.k_min()});

        auto block_size_f = [](int_t total, int_t block_size, int_t block_no) {
            auto n = (total + block_size - 1) / block_size;
            return block_no == n - 1 ? total - block_no * block_size : block_size;
        };
        auto total_i = grid.i_high_bound() - grid.i_low_bound() + 1;
        auto total_j = grid.j_high_bound() - grid.j_low_bound() + 1;
        int_t size_i = block_size_f(total_i, block_i_size(backend_target), execution_info.bi) + extent_t::iplus::value -
                       extent_t::iminus::value;
        int_t size_j = block_size_f(total_j, block_j_size(backend_target), execution_info.bj) + extent_t::jplus::value -
                       extent_t::jminus::value;
        static constexpr int_t n_colors =
            _impl_mss_loop_x86::get_ncolors<typename RunFunctorArgs::loop_intervals_t>::value;
        for (int_t i = 0; i != size_i; ++i) {
            gridtools::for_each<meta::make_indices_c<n_colors>>(
                _impl_mss_loop_x86::color_execution_functor<RunFunctorArgs, iterate_domain_t, Grid>{
                    it_domain, grid, size_j});
            it_domain.increment_c(integral_constant<int, -n_colors>{});
            it_domain.increment_i();
        }
    }
} // namespace gridtools

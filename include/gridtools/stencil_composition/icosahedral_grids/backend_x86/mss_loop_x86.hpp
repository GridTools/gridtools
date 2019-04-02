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
#include "../../backend_x86/basic_token_execution_x86.hpp"
#include "../../iteration_policy.hpp"
#include "../../pos3.hpp"
#include "../esf_metafunctions.hpp"
#include "../stage.hpp"
#include "./iterate_domain_x86.hpp"
#include "./run_esf_functor_x86.hpp"

/**@file
 * @brief mss loop implementations for the x86 backend
 */
namespace gridtools {
    namespace _impl_mss_loop_x86 {
        template <size_t Color>
        struct loop_interval_contains_color {
            template <class T>
            GT_META_DEFINE_ALIAS(apply,
                meta::any_of,
                (stage_group_contains_color<Color>::template apply, GT_META_CALL(meta::at_c, (T, 2))));
        };

        /**
         * @tparam RunFunctorArgs run functor argument type with the main configuration of the MSS
         * @tparam IterateDomain iterator domain class
         * @tparam Grid grid object as it provided by user.
         */
        template <typename RunFunctorArgs, typename IterateDomain, typename Grid>
        struct color_execution_functor {
          private:
            GT_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value), GT_INTERNAL_ERROR);
            GT_STATIC_ASSERT((is_iterate_domain<IterateDomain>::value), GT_INTERNAL_ERROR);
            GT_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);

            template <class Color>
            GT_META_DEFINE_ALIAS(has_color,
                meta::any_of,
                (loop_interval_contains_color<Color::value>::template apply,
                    typename RunFunctorArgs::loop_intervals_t));

            IterateDomain &m_it_domain;
            Grid const &m_grid;
            uint_t m_loop_size;

          public:
            color_execution_functor(IterateDomain &it_domain, Grid const &grid, uint_t loop_size)
                : m_it_domain(it_domain), m_grid(grid), m_loop_size(loop_size) {}

            template <class Color, enable_if_t<has_color<Color>::value, int> = 0>
            void operator()(Color) const {
                for (uint_t j = 0; j != m_loop_size; ++j) {
                    auto memorized_index = m_it_domain.index();
                    run_functors_on_interval<RunFunctorArgs, run_esf_functor_x86<Color::value>>(m_it_domain, m_grid);
                    m_it_domain.set_index(memorized_index);
                    m_it_domain.increment_j();
                }
                m_it_domain.increment_j(-m_loop_size);
                m_it_domain.increment_c();
            }
            template <class Color, enable_if_t<!has_color<Color>::value, int> = 0>
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

        using iterate_domain_arguments_t = iterate_domain_arguments<backend::x86,
            LocalDomain,
            typename RunFunctorArgs::esf_sequence_t,
            typename RunFunctorArgs::cache_sequence_t,
            Grid>;
        using iterate_domain_t = iterate_domain_x86<iterate_domain_arguments_t>;
        iterate_domain_t it_domain(local_domain);

        using extent_t = GT_META_CALL(get_extent_from_loop_intervals, typename RunFunctorArgs::loop_intervals_t);
        using interval_t = GT_META_CALL(meta::first, typename RunFunctorArgs::loop_intervals_t);
        using from_t = GT_META_CALL(meta::first, interval_t);
        it_domain.initialize({grid.i_low_bound(), grid.j_low_bound(), grid.k_min()},
            {execution_info.bi, execution_info.bj, 0},
            {extent_t::iminus::value,
                extent_t::jminus::value,
                static_cast<int_t>(grid.template value_at<from_t>() - grid.k_min())});

        auto block_size_f = [](uint_t total, uint_t block_size, uint_t block_no) {
            auto n = (total + block_size - 1) / block_size;
            return block_no == n - 1 ? total - block_no * block_size : block_size;
        };
        auto total_i = grid.i_high_bound() - grid.i_low_bound() + 1;
        auto total_j = grid.j_high_bound() - grid.j_low_bound() + 1;
        const uint_t size_i = block_size_f(total_i, block_i_size(backend_target), execution_info.bi) +
                              extent_t::iplus::value - extent_t::iminus::value;
        const uint_t size_j = block_size_f(total_j, block_j_size(backend_target), execution_info.bj) +
                              extent_t::jplus::value - extent_t::jminus::value;
        using location_type_t = typename extract_esf_location_type<typename RunFunctorArgs::esf_sequence_t>::type;
        static constexpr int_t n_colors = location_type_t::n_colors::value;
        for (uint_t i = 0; i != size_i; ++i) {
            gridtools::for_each<GT_META_CALL(meta::make_indices_c, n_colors)>(
                _impl_mss_loop_x86::color_execution_functor<RunFunctorArgs, iterate_domain_t, Grid>{
                    it_domain, grid, size_j});
            it_domain.increment_c(integral_constant<int, -n_colors>{});
            it_domain.increment_i();
        }
    }
} // namespace gridtools

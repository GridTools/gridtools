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
#include "../../../meta.hpp"
#include "../../backend_naive/basic_token_execution_naive.hpp"
#include "../../iteration_policy.hpp"
#include "../../loop_interval.hpp"
#include "../../pos3.hpp"
#include "../esf_metafunctions.hpp"
#include "../stage.hpp"
#include "iterate_domain_naive.hpp"

/**@file
 * @brief mss loop implementations for the naive backend
 */
namespace gridtools {
    namespace _impl_mss_loop_naive {
        template <size_t Color>
        struct loop_interval_contains_color {
            template <class T>
            GT_META_DEFINE_ALIAS(apply,
                meta::any_of,
                (stage_group_contains_color<Color>::template apply, GT_META_CALL(meta::at_c, (T, 2))));
        };

        template <uint_t Color>
        struct run_esf_functor_naive {
            template <class StageGroups, class ItDomain>
            static void exec(ItDomain &it_domain) {
                using stages_t = GT_META_CALL(meta::flatten, StageGroups);
                GT_STATIC_ASSERT(meta::length<stages_t>::value == 1, GT_INTERNAL_ERROR);
                using stage_t = GT_META_CALL(meta::first, stages_t);
                stage_t::template exec<Color>(it_domain);
            }
        };

        template <class LoopIntervals>
        struct get_ncolors;

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
                    run_functors_on_interval<RunFunctorArgs, run_esf_functor_naive<Color::value>>(m_it_domain, m_grid);
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

    } // namespace _impl_mss_loop_naive

    /**
     * @brief main execution of a mss. Defines the IJ loop bounds of this particular block
     * and sequentially executes all the functors in the mss
     * @tparam RunFunctorArgs run functor arguments
     */
    template <class RunFunctorArgs, class LocalDomain, class Grid, class ExecutionInfo>
    GT_FORCE_INLINE static void mss_loop(
        backend::naive const &, LocalDomain const &local_domain, Grid const &grid, const ExecutionInfo &) {
        GT_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((is_local_domain<LocalDomain>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);

        using iterate_domain_arguments_t =
            iterate_domain_arguments<backend::naive, LocalDomain, typename RunFunctorArgs::esf_sequence_t>;
        using iterate_domain_t = iterate_domain_naive<iterate_domain_arguments_t>;

        iterate_domain_t it_domain(local_domain);

        using extent_t = GT_META_CALL(get_extent_from_loop_intervals, typename RunFunctorArgs::loop_intervals_t);
        using interval_t = GT_META_CALL(meta::first, typename RunFunctorArgs::loop_intervals_t);
        using from_t = GT_META_CALL(meta::first, interval_t);
        it_domain.initialize({grid.i_low_bound(), grid.j_low_bound(), grid.k_min()},
            {0, 0, 0},
            {extent_t::iminus::value,
                extent_t::jminus::value,
                static_cast<int_t>(grid.template value_at<from_t>() - grid.k_min())});

        const uint_t size_i =
            grid.i_high_bound() - grid.i_low_bound() + 1 + extent_t::iplus::value - extent_t::iminus::value;
        const uint_t size_j =
            grid.j_high_bound() - grid.j_low_bound() + 1 + extent_t::jplus::value - extent_t::jminus::value;
        static constexpr int_t n_colors =
            _impl_mss_loop_naive::get_ncolors<typename RunFunctorArgs::loop_intervals_t>::value;
        for (uint_t i = 0; i != size_i; ++i) {
            gridtools::for_each<GT_META_CALL(meta::make_indices_c, n_colors)>(
                _impl_mss_loop_naive::color_execution_functor<RunFunctorArgs, iterate_domain_t, Grid>{
                    it_domain, grid, size_j});
            it_domain.increment_c(integral_constant<int, -n_colors>{});
            it_domain.increment_i();
        }
    }
} // namespace gridtools

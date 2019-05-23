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
#include <utility>

#include "../../../common/defs.hpp"
#include "../../../common/generic_metafunctions/for_each.hpp"
#include "../../../common/host_device.hpp"
#include "../../../common/integral_constant.hpp"
#include "../../../meta.hpp"
#include "../../execution_types.hpp"
#include "../../iterate_domain_aux.hpp"
#include "../../local_domain.hpp"
#include "../../run_functor_arguments.hpp"
#include "../../sid/concept.hpp"
#include "../../sid/loop.hpp"
#include "../dim.hpp"
#include "../grid.hpp"

/**@file
 * @brief mss loop implementations for the x86 backend
 */
namespace gridtools {
    namespace mss_loop_x86_impl_ {
        template <class ExecutionType,
            class From,
            class To,
            class StageGroups,
            class Grid,
            class Ptr,
            class Strides,
            std::enable_if_t<!meta::is_empty<StageGroups>::value, int> = 0>
        GT_FORCE_INLINE void execute_interval(
            loop_interval<From, To, StageGroups>, Grid const &grid, Ptr &ptr, Strides const &strides) {
            int_t n = grid.count(From{}, To{});
            for (int_t i = 0; i < n; ++i) {
                for_each<meta::flatten<StageGroups>>([&](auto stage) { stage(ptr, strides); });
                sid::shift(ptr, sid::get_stride<dim::k>(strides), execute::step<ExecutionType>);
            }
        }

        template <class ExecutionType,
            class Level,
            class StageGroups,
            class Grid,
            class Ptr,
            class Strides,
            std::enable_if_t<!meta::is_empty<StageGroups>::value, int> = 0>
        GT_FORCE_INLINE void execute_interval(
            loop_interval<Level, Level, StageGroups>, Grid const &grid, Ptr &ptr, Strides const &strides) {
            for_each<meta::flatten<StageGroups>>([&](auto stage) { stage(ptr, strides); });
            sid::shift(ptr, sid::get_stride<dim::k>(strides), execute::step<ExecutionType>);
        }

        template <class ExecutionType,
            class From,
            class To,
            class StageGroups,
            class Grid,
            class Ptr,
            class Strides,
            std::enable_if_t<meta::is_empty<StageGroups>::value, int> = 0>
        GT_FORCE_INLINE void execute_interval(
            loop_interval<From, To, StageGroups>, Grid const &grid, Ptr &ptr, Strides const &strides) {
            sid::shift(ptr, sid::get_stride<dim::k>(strides), execute::step<ExecutionType> * grid.count(From{}, To{}));
        }
    } // namespace mss_loop_x86_impl_

    /**
     * @brief main execution of a mss. Defines the IJ loop bounds of this particular block
     * and sequentially executes all the functors in the mss
     * @tparam RunFunctorArgs run functor arguments
     */
    template <class RunFunctorArgs, class LocalDomain, class Grid, class ExecutionInfo>
    GT_FORCE_INLINE void mss_loop(
        backend::x86, LocalDomain const &local_domain, Grid const &grid, const ExecutionInfo &execution_info) {
        GT_STATIC_ASSERT(is_run_functor_arguments<RunFunctorArgs>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(is_local_domain<LocalDomain>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);

        using loop_intervals_t = typename RunFunctorArgs::loop_intervals_t;
        using execution_type_t = typename RunFunctorArgs::execution_type_t;
        using extent_t = get_extent_from_loop_intervals<loop_intervals_t>;
        using from_t = meta::first<meta::first<loop_intervals_t>>;
        using to_t = meta::second<meta::last<loop_intervals_t>>;

        int_t k_first = grid.template value_at<from_t>();

        auto ptr = local_domain.m_ptr_holder();

        for_each_type<typename LocalDomain::esf_args_t>(
            initialize_index<backend::x86, LocalDomain>(local_domain.m_strides,
                {grid.i_low_bound(), grid.j_low_bound(), grid.k_min()},
                {execution_info.bi, execution_info.bj, 0},
                {extent_t::iminus::value, extent_t::jminus::value, k_first - grid.k_min()},
                ptr));

        auto block_size_f = [](int_t total, int_t block_size, int_t block_no) {
            int_t n = (total + block_size - 1) / block_size;
            return block_no == n - 1 ? total - block_no * block_size : block_size;
        };
        int_t total_i = grid.i_high_bound() - grid.i_low_bound() + 1;
        int_t total_j = grid.j_high_bound() - grid.j_low_bound() + 1;
        int_t size_i = block_size_f(total_i, block_i_size(backend::x86{}), execution_info.bi) + extent_t::iplus::value -
                       extent_t::iminus::value;
        int_t size_j = block_size_f(total_j, block_j_size(backend::x86{}), execution_info.bj) + extent_t::jplus::value -
                       extent_t::jminus::value;

        auto i_loop = sid::make_loop<dim::i>(size_i);
        auto j_loop = sid::make_loop<dim::j>(size_j);
        auto k_loop = [&](auto ptr, auto const &strides) {
            for_each<loop_intervals_t>([&](auto loop_interval) {
                mss_loop_x86_impl_::execute_interval<execution_type_t>(loop_interval, grid, ptr, strides);
            });
        };

        i_loop(j_loop(k_loop))(ptr, local_domain.m_strides);
    }
} // namespace gridtools

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
#include "../../../common/tuple.hpp"
#include "../../../common/tuple_util.hpp"
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
        template <class Ptr, class Strides>
        struct invoke_f {
            Ptr &m_ptr;
            Strides const &m_strides;
            template <class Fun>
            GT_FORCE_INLINE void operator()(Fun &&fun) const {
                fun(m_ptr, m_strides);
            }
        };

        template <class Step, class Count, class Stages, bool = meta::is_empty<Stages>::value>
        struct interval_executor_f {
            Count m_count;

            template <class Ptr, class Strides>
            GT_FORCE_INLINE void operator()(Ptr ptr, Strides const &strides) const {
                for (int_t i = 0; i < m_count; ++i) {
                    for_each<Stages>(invoke_f<Ptr, Strides>{ptr, strides});
                    sid::shift(ptr, sid::get_stride<dim::k>(strides), Step{});
                }
            }
        };

        template <class Step, class Count, class Stages>
        struct interval_executor_f<Step, Count, Stages, true> {
            decltype(Count{} * Step{}) m_offset;

            interval_executor_f(Count count) : m_offset(count * Step{}) {}

            template <class Ptr, class Strides>
            GT_FORCE_INLINE void operator()(Ptr ptr, Strides const &strides) const {
                sid::shift(ptr, sid::get_stride<dim::k>(strides), m_offset);
            }
        };

        template <class Step, class Stages>
        struct interval_executor_f<Step, integral_constant<int_t, 1>, Stages, false> {

            interval_executor_f(integral_constant<int_t, 1>) {}

            template <class Ptr, class Strides>
            GT_FORCE_INLINE void operator()(Ptr ptr, Strides const &strides) const {
                for_each<Stages>(invoke_f<Ptr, Strides>{ptr, strides});
                sid::shift(ptr, sid::get_stride<dim::k>(strides), Step{});
            }
        };

        template <class FromLevel, class ToLevel, class Grid>
        using count_type = decltype(std::declval<Grid const>().count(FromLevel{}, ToLevel{}));

        template <class ExecutionType, class Grid>
        struct executor_maker_f {
            using step_t = conditional_t<execute::is_backward<ExecutionType>::value,
                integral_constant<int_t, -1>,
                integral_constant<int_t, 1>>;

            Grid const &m_grid;

            template <class LoopInterval,
                class From = GT_META_CALL(meta::first, LoopInterval),
                class To = GT_META_CALL(meta::second, LoopInterval),
                class Stages = GT_META_CALL(meta::flatten, GT_META_CALL(meta::third, LoopInterval))>
            interval_executor_f<step_t, count_type<From, To, Grid>, Stages> operator()(LoopInterval) const {
                return {m_grid.count(From{}, To{})};
            }
        };

        template <class Executors>
        struct k_loop_f {
            Executors m_executors;

            template <class Ptr, class Strides>
            GT_FORCE_INLINE void operator()(Ptr ptr, Strides const &strides) const {
                tuple_util::for_each(invoke_f<Ptr, Strides>{ptr, strides}, m_executors);
            }
        };
        template <class Executors>
        k_loop_f<Executors> make_k_loop_raw(Executors executors) {
            return {std::move(executors)};
        }

        template <class LoopIntervals,
            class ExecutionType,
            class Grid,
            class Tuple = GT_META_CALL(meta::rename, (meta::ctor<tuple<>>::apply, LoopIntervals))>
        auto make_k_loop(Grid const &grid) GT_AUTO_RETURN(
            make_k_loop_raw(tuple_util::transform(executor_maker_f<ExecutionType, Grid>{grid}, Tuple{})));
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
        using extent_t = GT_META_CALL(get_extent_from_loop_intervals, loop_intervals_t);
        using from_t = GT_META_CALL(meta::first, GT_META_CALL(meta::first, loop_intervals_t));
        using to_t = GT_META_CALL(meta::second, GT_META_CALL(meta::last, loop_intervals_t));

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
        auto k_loop = mss_loop_x86_impl_::make_k_loop<loop_intervals_t, execution_type_t>(grid);

        i_loop(j_loop(k_loop))(ptr, local_domain.m_strides);
    }
} // namespace gridtools

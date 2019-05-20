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

#include "../common/defs.hpp"
#include "../meta.hpp"
#include "./execution_types.hpp"
#include "./extent.hpp"
#include "./level.hpp"

namespace gridtools {
    /**
     * A helper structure that holds an information specific to the so called loop interval.
     *
     * Loop interval is limited by its From and To interval levels.
     * From level means the level from what iteration along k-axis should start. It can be upper than ToLevel
     * if the execution direction is backward.
     *
     * It is assumed that for any elementary functor within the computation at most one apply overload is used for all
     * points in this interval. In other words each elementary functor could be bound to a single interval.
     *
     * @tparam FromLevel interval level where the execution should start
     * @tparam ToLevel interval level where the execution should end
     * @tparam Payload extra compile time info
     */
    template <class FromLevel, class ToLevel, class Payload>
    struct loop_interval {
        GT_STATIC_ASSERT(is_level<FromLevel>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(is_level<ToLevel>::value, GT_INTERNAL_ERROR);

        using type = loop_interval;
    };

    template <class T>
    GT_META_DEFINE_ALIAS(is_loop_interval, meta::is_instantiation_of, (loop_interval, T));

    namespace loop_interval_impl_ {
        GT_META_LAZY_NAMESPACE {
            template <class>
            struct reverse_loop_interval;
            template <class From, class To, class Payload>
            struct reverse_loop_interval<loop_interval<From, To, Payload>> {
                using type = loop_interval<To, From, Payload>;
            };
        }
        GT_META_DELEGATE_TO_LAZY(reverse_loop_interval, class T, T);

        template <class Stage>
        GT_META_DEFINE_ALIAS(get_extent_from_stage, meta::id, typename Stage::extent_t);

        template <class Interval,
            class StageGroups = meta::at_c<Interval, 2>,
            class Stages = meta::flatten<StageGroups>>
        GT_META_DEFINE_ALIAS(get_extents_from_interval, meta::transform, (get_extent_from_stage, Stages));

        template <class LoopIntervals, class ExtentsList = meta::transform<get_extents_from_interval, LoopIntervals>>
        GT_META_DEFINE_ALIAS(all_extents_of_loop_intervals, meta::flatten, ExtentsList);

    } // namespace loop_interval_impl_

    GT_META_LAZY_NAMESPACE {
        template <class Execute, class LoopIntervals>
        struct order_loop_intervals : meta::lazy::id<LoopIntervals> {};

        template <class LoopIntervals>
        struct order_loop_intervals<execute::backward, LoopIntervals> {
            using type = meta::reverse<meta::transform<loop_interval_impl_::reverse_loop_interval, LoopIntervals>>;
        };

        template <class LoopIntervals,
            class Extents = loop_interval_impl_::all_extents_of_loop_intervals<LoopIntervals>>
        struct get_extent_from_loop_intervals : meta::lazy::first<Extents> {
            GT_STATIC_ASSERT(meta::all_are_same<Extents>::value, GT_INTERNAL_ERROR);
            GT_STATIC_ASSERT((meta::all_of<is_extent, Extents>::value), GT_INTERNAL_ERROR);
        };

        template <class LoopIntervals, template <class...> class L>
        struct get_extent_from_loop_intervals<LoopIntervals, L<>> : meta::lazy::id<extent<>> {};
    }
    /**
     * Applies execution policy to the list of loop intervals.
     * For backward execution loop_intervals are reversed and for each interval From and To levels got swapped.
     */
    GT_META_DELEGATE_TO_LAZY(order_loop_intervals, (class Execute, class LoopIntervals), (Execute, LoopIntervals));

    GT_META_DELEGATE_TO_LAZY(get_extent_from_loop_intervals, class LoopIntervals, LoopIntervals);

} // namespace gridtools

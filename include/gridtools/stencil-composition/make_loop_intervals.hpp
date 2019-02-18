/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "../meta.hpp"
#include "./interval.hpp"
#include "./level.hpp"
#include "./loop_interval.hpp"

namespace gridtools {
    namespace _impl {
        template <class From>
        struct make_level_index {
            GT_STATIC_ASSERT(is_level_index<From>::value, GT_INTERNAL_ERROR);
            template <class N>
            GT_META_DEFINE_ALIAS(apply, level_index, (N::value + From::value, From::offset_limit));
        };

        template <class Acc, class Cur, class Prev = GT_META_CALL(meta::last, Acc)>
        GT_META_DEFINE_ALIAS(loop_level_inserter,
            meta::if_,
            (std::is_same<GT_META_CALL(meta::second, Cur), GT_META_CALL(meta::second, Prev)>,
                Acc,
                GT_META_CALL(meta::push_back, (Acc, Cur))));

        template <class LoopLevel, class From = GT_META_CALL(meta::first, LoopLevel)>
        GT_META_DEFINE_ALIAS(get_previous_to, level_index, (From::value - 1, From::offset_limit));

        template <class LoopLevel, class ToIndex, class FromIndex = GT_META_CALL(meta::first, LoopLevel)>
        GT_META_DEFINE_ALIAS(make_loop_interval,
            loop_interval,
            (GT_META_CALL(index_to_level, FromIndex),
                GT_META_CALL(index_to_level, ToIndex),
                GT_META_CALL(meta::second, LoopLevel)));

        template <class T>
        struct has_stages : negation<meta::is_empty<GT_META_CALL(meta::at_c, (T, 2))>> {};
    } // namespace _impl

    GT_META_LAZY_NAMESPACE {
        template <template <class...> class StagesMaker, class Interval>
        struct make_loop_intervals {
            GT_STATIC_ASSERT(is_interval<Interval>::value, GT_INTERNAL_ERROR);

            // produce the list of all level_indices that the give interval has
            using from_index_t = GT_META_CALL(level_to_index, typename Interval::FromLevel);
            using to_index_t = GT_META_CALL(level_to_index, typename Interval::ToLevel);
            using nums_t = GT_META_CALL(meta::make_indices_c, to_index_t::value - from_index_t::value + 1);
            using indices_t = GT_META_CALL(
                meta::transform, (_impl::make_level_index<from_index_t>::template apply, nums_t));

            // produce stages and zip them with indices
            using stages_t = GT_META_CALL(meta::transform, (StagesMaker, indices_t));
            using all_loop_levels_t = GT_META_CALL(meta::zip, (indices_t, stages_t));

            GT_STATIC_ASSERT(!meta::is_empty<all_loop_levels_t>::value, GT_INTERNAL_ERROR);

            // merge the sequential levels that have the same stages together
            using first_of_all_loop_levels_t = GT_META_CALL(meta::first, all_loop_levels_t);
            using rest_of_all_loop_levels_t = GT_META_CALL(meta::pop_front, all_loop_levels_t);
            using loop_levels_t = GT_META_CALL(meta::lfold,
                (_impl::loop_level_inserter, meta::list<first_of_all_loop_levels_t>, rest_of_all_loop_levels_t));

            GT_STATIC_ASSERT(!meta::is_empty<loop_levels_t>::value, GT_INTERNAL_ERROR);

            // calculate the to_indices
            using rest_loop_levels_t = GT_META_CALL(meta::pop_front, loop_levels_t);
            using intermediate_to_indices_t = GT_META_CALL(
                meta::transform, (_impl::get_previous_to, rest_loop_levels_t));
            using to_indices_t = GT_META_CALL(meta::push_back, (intermediate_to_indices_t, to_index_t));

            // make loop intervals
            using loop_intervals_t = GT_META_CALL(
                meta::transform, (_impl::make_loop_interval, loop_levels_t, to_indices_t));

            // filter out interval with the empty stages
            using type = GT_META_CALL(meta::filter, (_impl::has_stages, loop_intervals_t));
        };
    }
    /**
     * Calculate the loop intervals together with the stages that should be executed within each of them.
     *
     * @tparam StageMaker - a meta calllback that gets stages that shuld be executed for the given level_index.
     *                      It should take level_index and return a type list of something. [In our design this
     *                      "something" is a list of list of Stages, but `make_loop_intervals` only uses the fact that
     *                      it is a type list] An empty list means that there is nothing to execute for the given level.
     * @tparam Interval   - the interval from the user that represents the requested computation area in k-direction
     *
     *  The algorithm:
     *   - for each level within the interval compute stages using provided callback.
     *   - use lfold to glue together the sequential levels that have the same calculated stages.
     *   - transform the result into the sequence of loop intervals
     *   - filter out the intervals with empty stages
     *
     *   TODO(anstaf): verify that doxy formatting is OK here.
     */
    GT_META_DELEGATE_TO_LAZY(
        make_loop_intervals, (template <class...> class StagesMaker, class Interval), (StagesMaker, Interval));

} // namespace gridtools

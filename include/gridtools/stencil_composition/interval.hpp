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
#include "../common/host_device.hpp"
#include "../meta.hpp"
#include "level.hpp"

namespace gridtools {

    namespace _impl {
        constexpr int_t sign(int_t value) { return (0 < value) - (value < 0); }

        constexpr int_t add_offset(int_t offset, int_t value) {
            return sign(offset + value) == sign(offset) ? offset + value : offset + value + sign(value);
        }
    } // namespace _impl

    /**
     * @struct Interval
     * Structure defining a closed interval on an axis given two levels
     */
    template <class TFromLevel, class TToLevel>
    struct interval {
        // HACK allow implicit conversion from the from level to any interval starting with the from level
        // (due to this trick we can search all do method overloads starting at a given from position)
        interval() = default;
        interval(TFromLevel);

        // check the parameters are of type level
        static_assert(is_level<TFromLevel>::value, "check the first template parameter is of type level");
        static_assert(is_level<TToLevel>::value, "check the second template parameter is of type level");

        // check the from level is lower or equal to the to level
        static_assert(TFromLevel::splitter < TToLevel::splitter ||
                          (TFromLevel::splitter == TToLevel::splitter && TFromLevel::offset <= TToLevel::offset),
            "check the from level is lower or equal to the to level");
        static_assert(TFromLevel::offset_limit == TToLevel::offset_limit, "levels must have same offset limit");
        static constexpr int_t offset_limit = TFromLevel::offset_limit;

        // define the from and to splitter indexes
        typedef TFromLevel FromLevel;
        typedef TToLevel ToLevel;

        // User API: helper to access the first and last level as an interval
        using first_level = interval<TFromLevel, TFromLevel>;
        using last_level = interval<TToLevel, TToLevel>;

        /**
         * @brief returns an interval where the boundaries are modified according to left and right
         * @param left moves the left boundary, the interval is enlarged (left < 0) or shrunk (left > 0)
         * @param right moves the right boundary, the interval is enlarged (right > 0) or shrunk (right < 0)
         */
        template <int_t left, int_t right>
        struct modify_impl {
            static_assert((_impl::add_offset(TFromLevel::offset, left) >= -TFromLevel::offset_limit &&
                              _impl::add_offset(TToLevel::offset, right) <= TFromLevel::offset_limit),
                "You are trying to modify an interval to increase beyond its maximal offset.");
            static_assert(TFromLevel::splitter < TToLevel::splitter ||
                              _impl::add_offset(TFromLevel::offset, left) <= _impl::add_offset(TToLevel::offset, right),
                "You are trying to modify an interval such that the result is an empty interval(left boundary > right "
                "boundary).");
            using type = interval<
                level<TFromLevel::splitter, _impl::add_offset(TFromLevel::offset, left), TFromLevel::offset_limit>,
                level<TToLevel::splitter, _impl::add_offset(TToLevel::offset, right), TToLevel::offset_limit>>;
        };
        template <int_t left, int_t right>
        using modify = typename modify_impl<left, right>::type;
        template <int_t dir>
        using shift = modify<dir, dir>;
    };

    /**
     * @struct is_interval
     * Trait returning true it the template parameter is an interval
     */
    template <class T>
    using is_interval = meta::is_instantiation_of<interval, T>;

    /**
     * @struct interval_from_index
     * Meta function returning the interval from level index
     */
    template <class Index, class Level = index_to_level<Index>>
    using interval_from_index = interval<Level, Level>;

    /**
     * @struct interval_to_index
     * Meta function returning the interval to level index
     */
    template <class>
    struct interval_to_index;

    template <class TFromLevel, class TToLevel>
    struct interval_to_index<interval<TFromLevel, TToLevel>> : level_to_index<TToLevel> {};

    /**
     * @struct make_interval
     * Meta function computing an interval given a from and a to level index
     */
    template <class FromIndex, class ToIndex>
    using make_interval = interval<index_to_level<FromIndex>, index_to_level<ToIndex>>;

    namespace lazy {
        template <class...>
        struct concat_intervals;

        template <class T>
        struct concat_intervals<T> {
            using type = T;
        };
        template <class From, class Level, class NextLevel, class To>
        struct concat_intervals<interval<From, Level>, interval<NextLevel, To>> {
            static_assert(level_to_index<Level>::value + 1 == level_to_index<NextLevel>::value, GT_INTERNAL_ERROR);
            using type = interval<From, To>;
        };
        template <class... Intervals>
        struct concat_intervals {
            using type = meta::combine<meta::force<lazy::concat_intervals>::apply, meta::list<Intervals...>>;
        };

        template <class...>
        struct enclosing_interval;

        template <class T>
        struct enclosing_interval<T> {
            using type = T;
        };
        template <class LFrom, class LTo, class RFrom, class RTo>
        struct enclosing_interval<interval<LFrom, LTo>, interval<RFrom, RTo>> {
            using from_t = meta::if_c<(level_to_index<LFrom>::value < level_to_index<RFrom>::value), LFrom, RFrom>;
            using to_t = meta::if_c<(level_to_index<LTo>::value > level_to_index<RTo>::value), LTo, RTo>;
            using type = interval<from_t, to_t>;
        };
        template <class... Intervals>
        struct enclosing_interval {
            using type = meta::combine<meta::force<lazy::enclosing_interval>::apply, meta::list<Intervals...>>;
        };
    } // namespace lazy
    GT_META_DELEGATE_TO_LAZY(concat_intervals, class... Ts, Ts...);
    GT_META_DELEGATE_TO_LAZY(enclosing_interval, class... Ts, Ts...);
} // namespace gridtools

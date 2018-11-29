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

#include "../common/defs.hpp"
#include "../common/generic_metafunctions/meta.hpp"
#include "../common/host_device.hpp"
#include "./level.hpp"

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
        GT_FUNCTION interval(TFromLevel);

        using type = interval;

        // check the parameters are of type level
        GRIDTOOLS_STATIC_ASSERT(is_level<TFromLevel>::value, "check the first template parameter is of type level");
        GRIDTOOLS_STATIC_ASSERT(is_level<TToLevel>::value, "check the second template parameter is of type level");

        // check the from level is lower or equal to the to level
        GRIDTOOLS_STATIC_ASSERT(
            (TFromLevel::splitter < TToLevel::splitter) ||
                (TFromLevel::splitter == TToLevel::splitter && TFromLevel::offset <= TToLevel::offset),
            "check the from level is lower or equal to the to level");
        GRIDTOOLS_STATIC_ASSERT(
            TFromLevel::offset_limit == TToLevel::offset_limit, "levels must have same offset limit");
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
            GRIDTOOLS_STATIC_ASSERT((_impl::add_offset(TFromLevel::offset, left) >= -TFromLevel::offset_limit &&
                                        _impl::add_offset(TToLevel::offset, right) <= TFromLevel::offset_limit),
                "You are trying to modify an interval to increase beyond its maximal offset.");
            GRIDTOOLS_STATIC_ASSERT(
                ((TFromLevel::splitter < TToLevel::splitter) ||
                    (_impl::add_offset(TFromLevel::offset, left) <= _impl::add_offset(TToLevel::offset, right))),
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
    GT_META_DEFINE_ALIAS(is_interval, meta::is_instantiation_of, (interval, T));

    /**
     * @struct interval_from_index
     * Meta function returning the interval from level index
     */
    template <class>
    struct interval_from_index;

    template <class TFromLevel, class TToLevel>
    struct interval_from_index<interval<TFromLevel, TToLevel>> : level_to_index<TFromLevel> {};

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
    GT_META_DEFINE_ALIAS(
        make_interval, interval, (GT_META_CALL(index_to_level, FromIndex), GT_META_CALL(index_to_level, ToIndex)));
} // namespace gridtools

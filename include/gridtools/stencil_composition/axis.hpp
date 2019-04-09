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
#include "..//common/defs.hpp"
#include "../common/array.hpp"
#include "../common/generic_metafunctions/accumulate.hpp"
#include "../common/generic_metafunctions/is_all_integrals.hpp"
#include "../common/variadic_pack_metafunctions.hpp"
#include "interval.hpp"
#include "level.hpp"
#include <type_traits>

namespace gridtools {

    namespace axis_config {
        template <int_t V>
        struct offset_limit : std::integral_constant<int_t, V> {};

        template <int_t V>
        struct extra_offsets : std::integral_constant<int_t, V> {};
    } // namespace axis_config

    /**
     * Defines an axis_interval_t which spans the whole axis.
     * @param NIntervals Number of intervals the axis should support
     * @param LevelOffsetLimit Maximum offset relative to the splitter position that is required to specify the
     * intervals
     * @param (non-API) ExtraOffsetsAroundFullInterval Special case when access of k-values around the full_interval
     * (i.e. after the last or before the first splitter value) are needed. (Note that the default interval will span
     * the whole axis_interval_t.)
     */
    template <size_t, class = axis_config::offset_limit<2>, class = axis_config::extra_offsets<0>>
    class axis;

    template <size_t NIntervals, int_t LevelOffsetLimit, int_t ExtraOffsetsAroundFullInterval>
    class axis<NIntervals,
        axis_config::offset_limit<LevelOffsetLimit>,
        axis_config::extra_offsets<ExtraOffsetsAroundFullInterval>> {
      private:
        template <size_t... IntervalIDs>
        struct interval_impl {
            GT_STATIC_ASSERT((is_continuous(IntervalIDs...)), "Intervals must be continuous.");
            static constexpr size_t min_id = constexpr_min(IntervalIDs...);
            static constexpr size_t max_id = constexpr_max(IntervalIDs...);
            GT_STATIC_ASSERT((max_id < NIntervals), "Interval ID out of bounds for this axis.");

            using type = interval<level<min_id, 1, LevelOffsetLimit>, level<max_id + 1, -1, LevelOffsetLimit>>;
        };

      public:
        using axis_interval_t =
            interval<level<0, _impl::add_offset(1, -ExtraOffsetsAroundFullInterval), LevelOffsetLimit>,
                level<NIntervals, _impl::add_offset(1, ExtraOffsetsAroundFullInterval), LevelOffsetLimit>>;

        using full_interval = interval<level<0, 1, LevelOffsetLimit>, level<NIntervals, -1, LevelOffsetLimit>>;

        template <typename... IntervalSizes,
            typename std::enable_if<sizeof...(IntervalSizes) == NIntervals && is_all_integral<IntervalSizes...>::value,
                int>::type = 0>
        axis(IntervalSizes... interval_sizes) : interval_sizes_{interval_sizes...} {}

        uint_t interval_size(const uint_t index) const { return interval_sizes_[index]; }
        const array<uint_t, NIntervals> &interval_sizes() const { return interval_sizes_; };

        template <size_t... IntervalIDs>
        using get_interval = typename interval_impl<IntervalIDs...>::type;

        static constexpr int_t level_offset_limit = LevelOffsetLimit;

      private:
        array<uint_t, NIntervals> interval_sizes_;
    };
} // namespace gridtools

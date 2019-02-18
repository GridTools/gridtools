/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#include "../common/array.hpp"
#include "../common/generic_metafunctions/accumulate.hpp"
#include "../common/generic_metafunctions/is_all_integrals.hpp"
#include "../common/variadic_pack_metafunctions.hpp"
#include "interval.hpp"
#include "level.hpp"

namespace gridtools {
    /**
     * Defines an axis_interval_t which is the former user-defined axis type and a full_interval which spans the whole
     * axis.
     * @param NIntervals Number of intervals the axis should support
     * @param ExtraOffsetsAroundFullInterval Special case when access of k-values around the full_interval (i.e. after
     * the last or before the first splitter value) are needed. (Note that the default interval will span the whole
     * axis_interval_t.)
     */
    template <size_t NIntervals, int_t ExtraOffsetsAroundFullInterval = 0, int_t LevelOffsetLimit = 2>
    class axis {
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

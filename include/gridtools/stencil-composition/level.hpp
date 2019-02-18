/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <type_traits>

#include "../common/defs.hpp"
#include "../meta/macros.hpp"

namespace gridtools {

    namespace _impl {
        constexpr int_t calc_level_index(uint_t splitter, int_t offset, int_t limit) {
            return limit * (2 * (int_t)splitter + 1) + offset - (offset >= 0);
        }
        constexpr int_t get_splitter_from_index(int_t index, int_t limit) { return index / (2 * limit); }
        constexpr int_t get_offset_from_index(int_t index, int_t limit) {
            return index % (2 * limit) - limit + (index % (2 * limit) >= limit);
        }
    } // namespace _impl

    /**
     * @struct Level
     * Structure defining an axis position relative to a splitter
     */
    template <uint_t Splitter, int_t Offset, int_t OffsetLimit>
    struct level {
        // check offset and splitter value ranges
        // (note that non negative splitter values simplify the index computation)
        GT_STATIC_ASSERT(Splitter >= 0 && Offset != 0, "check offset and splitter value ranges \n\
         (note that non negative splitter values simplify the index computation)");
        GT_STATIC_ASSERT(-OffsetLimit <= Offset && Offset <= OffsetLimit, "check offset and splitter value ranges \n\
         (note that non negative splitter values simplify the index computation)");

        // define splitter, level offset and offset limit
        static constexpr uint_t splitter = Splitter;
        static constexpr int_t offset = Offset;
        static constexpr int_t offset_limit = OffsetLimit;
        using type = level;
    };

    /**
     * @struct is_level
     * Trait returning true it the template parameter is a level
     */
    template <class>
    struct is_level : std::false_type {};

    template <uint_t Splitter, int_t Offset, int_t OffsetLimit>
    struct is_level<level<Splitter, Offset, OffsetLimit>> : std::true_type {};

    template <int_t Value, int_t OffsetLimit>
    struct level_index {
        static constexpr int_t value = Value;
        static constexpr int_t offset_limit = OffsetLimit;

        using type = level_index;
        using next = level_index<Value + 1, OffsetLimit>;
        using prior = level_index<Value - 1, OffsetLimit>;
    };

    template <class>
    struct is_level_index : std::false_type {};

    template <int_t Index, int_t OffsetLimit>
    struct is_level_index<level_index<Index, OffsetLimit>> : std::true_type {};

    template <class Level>
    GT_META_DEFINE_ALIAS(level_to_index,
        level_index,
        (_impl::calc_level_index(Level::splitter, Level::offset, Level::offset_limit), Level::offset_limit));

    /**
     * @struct index_to_level
     * Meta function converting a unique index back into a level
     */
    template <class Index>
    GT_META_DEFINE_ALIAS(index_to_level,
        level,
        (_impl::get_splitter_from_index(Index::value, Index::offset_limit),
            _impl::get_offset_from_index(Index::value, Index::offset_limit),
            Index::offset_limit));
} // namespace gridtools

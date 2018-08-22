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

#include <tuple>
#include <type_traits>

#include <boost/fusion/include/mpl.hpp>
#include <boost/fusion/include/std_tuple.hpp>

#include "../common/defs.hpp"
#include "../common/generic_metafunctions/meta.hpp"

namespace gridtools {

    namespace _impl {
        constexpr int_t calc_level_index(uint_t splitter, int_t offset, int_t limit) {
            return limit * (2 * splitter + 1) + offset - (offset >= 0);
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
        GRIDTOOLS_STATIC_ASSERT(Splitter >= 0 && Offset != 0, "check offset and splitter value ranges \n\
         (note that non negative splitter values simplify the index computation)");
        GRIDTOOLS_STATIC_ASSERT(
            -OffsetLimit <= Offset && Offset <= OffsetLimit, "check offset and splitter value ranges \n\
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

    /**
     * @struct make_range
     * Meta function converting two level indexes into a range
     */
    template <class FromIndex, class ToIndex>
    struct make_range {
        GRIDTOOLS_STATIC_ASSERT(
            is_level_index<FromIndex>::value, GT_INTERNAL_ERROR_MSG("metafunction input must be an index"));
        GRIDTOOLS_STATIC_ASSERT(
            is_level_index<ToIndex>::value, GT_INTERNAL_ERROR_MSG("metafunction input must be an index"));

        template <class Number>
        GT_META_DEFINE_ALIAS(to_level_index, level_index, (Number::value + FromIndex::value, FromIndex::offset_limit));

        using numbers_t = GT_META_CALL(meta::make_indices_c, ToIndex::value + 1 - FromIndex::value);
        using levels_t = GT_META_CALL(meta::transform, (to_level_index, numbers_t));
        using type = GT_META_CALL(meta::rename, (meta::ctor<std::tuple<>>::apply, levels_t));
    };
} // namespace gridtools

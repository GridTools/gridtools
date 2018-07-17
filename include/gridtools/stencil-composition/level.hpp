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
#include <boost/config.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/copy.hpp>
#include <boost/mpl/integral_c.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/static_assert.hpp>

namespace gridtools {
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
    };

    /**
     * @struct is_level
     * Trait returning true it the template parameter is a level
     */
    template <typename T>
    struct is_level : boost::mpl::false_ {};

    template <uint_t Splitter, int_t Offset, int_t OffsetLimit>
    struct is_level<level<Splitter, Offset, OffsetLimit>> : boost::mpl::true_ {};

    template <int_t Value, int_t OffsetLimit>
    struct level_index {
        static constexpr int_t value = Value;
        static constexpr int_t offset_limit = OffsetLimit;
        using type = level_index<Value, OffsetLimit>;

        using next = level_index<Value + 1, OffsetLimit>;
        using prior = level_index<Value - 1, OffsetLimit>;
    };

    template <typename T>
    struct is_level_index : boost::mpl::false_ {};

    template <int_t Index, int_t OffsetLimit>
    struct is_level_index<level_index<Index, OffsetLimit>> : boost::mpl::true_ {};

    /**
     * @struct level_to_index
     * Meta function computing a unique index given a level
     */
    template <typename Level>
    struct level_to_index {
      private:
        GRIDTOOLS_STATIC_ASSERT(is_level<Level>::value, GT_INTERNAL_ERROR_MSG("metafunction input must be a level"));

        static constexpr int_t splitter_index = Level::splitter;
        static constexpr int_t offset_index =
            (Level::offset < 0 ? Level::offset : Level::offset - 1) + Level::offset_limit;

      public:
        using type = level_index<2 * Level::offset_limit * splitter_index + offset_index, Level::offset_limit>;
    };

    /**
     * @struct index_to_level
     * Meta function converting a unique index back into a level
     */
    template <typename Index>
    struct index_to_level {
      private:
        GRIDTOOLS_STATIC_ASSERT(
            is_level_index<Index>::value, GT_INTERNAL_ERROR_MSG("metafunction input must be an index"));

        static constexpr uint_t splitter = Index::value / (2 * Index::offset_limit);
        static constexpr int_t offset_index = Index::value % (2 * Index::offset_limit) - Index::offset_limit;
        static constexpr int_t offset = offset_index < 0 ? offset_index : offset_index + 1;

      public:
        using type = level<splitter, offset, Index::offset_limit>;
    };

    /**
     * @struct make_range
     * Meta function converting two level indexes into a range
     */
    template <typename FromIndex, typename ToIndex>
    struct make_range {
      private:
        GRIDTOOLS_STATIC_ASSERT(
            is_level_index<FromIndex>::value, GT_INTERNAL_ERROR_MSG("metafunction input must be an index"));
        GRIDTOOLS_STATIC_ASSERT(
            is_level_index<ToIndex>::value, GT_INTERNAL_ERROR_MSG("metafunction input must be an index"));

        typedef boost::mpl::range_c<int_t, FromIndex::value, ToIndex::value + 1> range_type;
        typedef
            typename boost::mpl::copy<range_type, boost::mpl::back_inserter<boost::mpl::vector<>>>::type vector_type;
        struct index_wrap {
            template <typename Integer>
            struct apply {
                using type = level_index<Integer::value, FromIndex::offset_limit>;
            };
        };

      public:
        typedef typename boost::mpl::transform<vector_type, index_wrap>::type type;
    };

    template <uint_t F, int_t T, uint_t L>
    std::ostream &operator<<(std::ostream &s, level<F, T, L> const &) {
        return s << "(" << level<F, T, L>::Splitter::value << ", " << level<F, T, L>::Offset::value << ", "
                 << level<F, T, L>::offset_limit << ")";
    }
} // namespace gridtools

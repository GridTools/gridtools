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

#include <boost/static_assert.hpp>
#include <boost/mpl/integral_c.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/front.hpp>
#include <boost/mpl/back.hpp>
#include <boost/mpl/range_c.hpp>
#include "level.hpp"
#include "level_metafunctions.hpp"
#include "../common/host_device.hpp"
#include "../common/gt_assert.hpp"
#include "../common/generic_metafunctions/is_all.hpp"
#include "sfinae.hpp"

namespace gridtools {

    namespace internal {
        constexpr int_t add_offset(int_t offset, int_t value) {
            return (offset + value == 0) ? (offset + 2 * value) : (offset + value);
        }
    }

    /**
     * @struct Interval
     * Structure defining a closed interval on an axis given two levels
     */
    template < typename TFromLevel, typename TToLevel >
    struct interval {
        // HACK allow implicit conversion from the from level to any interval starting with the from level
        // (due to this trick we can search all do method overloads starting at a given from position)
        GT_FUNCTION
        interval(){};

        interval(sfinae::_impl::dummy_type) { assert(false); } // using this just for SFINAE

        GT_FUNCTION
        interval(TFromLevel){};

        static void static_info() {
            printf("level \"from\": splitter %d, offset %d \n", TFromLevel::Splitter::value, TFromLevel::Offset::value);
            printf("level \"to\": splitter %d, offset %d \n", TToLevel::Splitter::value, TToLevel::Offset::value);
        }
        // check the parameters are of type level
        GRIDTOOLS_STATIC_ASSERT(is_level< TFromLevel >::value, "check the first template parameter is of type level");
        GRIDTOOLS_STATIC_ASSERT(is_level< TToLevel >::value, "check the second template parameter is of type level");

        // check the from level is lower or equal to the to level
        GRIDTOOLS_STATIC_ASSERT((TFromLevel::Splitter::value < TToLevel::Splitter::value) ||
                                    (TFromLevel::Splitter::value == TToLevel::Splitter::value &&
                                        TFromLevel::Offset::value <= TToLevel::Offset::value),
            "check the from level is lower or equal to the to level");

        // define the from and to splitter indexes
        typedef TFromLevel FromLevel;
        typedef TToLevel ToLevel;

        // User API: helper to access the first and last level as an interval
        using first_level = interval< TFromLevel, TFromLevel >;
        using last_level = interval< TToLevel, TToLevel >;

        // User API: move bounds of the interval
        template < int_t left, int_t right >
        struct internal_modify_interval {
            GRIDTOOLS_STATIC_ASSERT((internal::add_offset(TFromLevel::Offset::value, left) >= -cLevelOffsetLimit &&
                                        internal::add_offset(TToLevel::Offset::value, right) <= cLevelOffsetLimit),
                "You are trying to modify an interval to increase beyond its maximal offset.");
            GRIDTOOLS_STATIC_ASSERT(((TFromLevel::Splitter::value < TToLevel::Splitter::value) ||
                                        (internal::add_offset(TFromLevel::Offset::value, left) <=
                                            internal::add_offset(TToLevel::Offset::value, right))),
                "You are trying to modify an interval such that the result is an empty interval(left boundary > right "
                "boundary).");
            using type =
                interval< level< TFromLevel::Splitter::value, internal::add_offset(TFromLevel::Offset::value, left) >,
                    level< TToLevel::Splitter::value, internal::add_offset(TToLevel::Offset::value, right) > >;
        };
        template < int_t left, int_t right >
        using modify = typename internal_modify_interval< left, right >::type;
        template < int_t dir >
        using shift = modify< dir, dir >;
    };

    /**
     * @struct is_interval
     * Trait returning true it the template parameter is an interval
     */
    template < typename T >
    struct is_interval : boost::mpl::false_ {};

    template < typename TFromLevel, typename TToLevel >
    struct is_interval< interval< TFromLevel, TToLevel > > : boost::mpl::true_ {};

    /**
     * @struct interval_from_index
     * Meta function returning the interval from level index
     */
    template < typename TInterval >
    struct interval_from_index;

    template < typename TFromLevel, typename TToLevel >
    struct interval_from_index< interval< TFromLevel, TToLevel > > : level_to_index< TFromLevel > {};

    /**
     * @struct interval_to_index
     * Meta function returning the interval to level index
     */
    template < typename TInterval >
    struct interval_to_index;

    template < typename TFromLevel, typename TToLevel >
    struct interval_to_index< interval< TFromLevel, TToLevel > > : level_to_index< TToLevel > {};

    /**
     * @struct make_interval
     * Meta function computing an interval given a from and a to level index
     */
    template < typename TFromIndex, typename ToIndex >
    struct make_interval {
        typedef interval< typename index_to_level< TFromIndex >::type, typename index_to_level< ToIndex >::type > type;
    };

    namespace internal {
        template < typename... Intervals >
        struct internal_join_interval {
            GRIDTOOLS_STATIC_ASSERT((is_all< is_interval, Intervals... >::value),
                GT_INTERNAL_ERROR_MSG("Expected all types to be intervals."));
            using from_levels_vector = sort_levels< typename Intervals::FromLevel... >;
            using to_levels_vector = sort_levels< typename Intervals::ToLevel... >;
            using type = interval< typename boost::mpl::back< from_levels_vector >::type,
                typename boost::mpl::front< to_levels_vector >::type >;
        };
    }
    /**
     * @brief returns interval which has all given intervals as subset
     */
    template < typename... Intervals >
    using join_interval = typename internal::internal_join_interval< Intervals... >::type;
} // namespace gridtools

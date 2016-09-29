/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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

#include <boost/mpl/vector.hpp>
#include <boost/mpl/front.hpp>
#include <boost/mpl/back.hpp>
#include <boost/mpl/sort.hpp>
#include "interval_metafunctions_fwd.hpp"

namespace gridtools {
#ifdef CXX11_ENABLED
    template < typename TInterval1 >
    struct check_interval {
        template < typename TInterval2, typename Enable = void >
        struct is_subset_of : boost::mpl::false_ {};
        template < typename TInterval2 >
        struct is_subset_of<
            TInterval2,
            typename std::enable_if<
                level_geq< typename TInterval1::FromLevel, typename TInterval2::FromLevel >::value &&
                level_leq< typename TInterval1::ToLevel, typename TInterval2::ToLevel >::value >::type >
            : boost::mpl::true_ {};

        template < typename TInterval2, typename Enable = void >
        struct is_strict_subset_of : boost::mpl::false_ {};
        template < typename TInterval2 >
        struct is_strict_subset_of<
            TInterval2,
            typename std::enable_if<
                level_gt< typename TInterval1::FromLevel, typename TInterval2::FromLevel >::value &&
                level_lt< typename TInterval1::ToLevel, typename TInterval2::ToLevel >::value >::type >
            : boost::mpl::true_ {};
    };

    /**
     * @struct join_interval_is_contiguous
     * Meta function to test if the union of two intervals is contiguous
     */
    template < typename TIntervalLeft, typename TIntervalRight, typename Enable = void >
    struct join_interval_is_contiguous : boost::mpl::false_ {};

    template < typename TIntervalLeft, typename TIntervalRight >
    struct join_interval_is_contiguous< TIntervalLeft,
        TIntervalRight,
        typename std::enable_if< (level_to_index< typename TIntervalLeft::ToLevel >::value + 1 >=
                                  level_to_index< typename TIntervalRight::FromLevel >::value) >::type >
        : boost::mpl::true_ {};

    template < typename TIntervalLeft, typename TIntervalRight >
    struct join_interval {
        GRIDTOOLS_STATIC_ASSERT(
            is_interval< TIntervalLeft >::value, "check the first template parameter is of type interval");
        GRIDTOOLS_STATIC_ASSERT(
            is_interval< TIntervalRight >::value, "check the second template parameter is of type interval");

        GRIDTOOLS_STATIC_ASSERT(
            (level_leq< typename TIntervalLeft::FromLevel, typename TIntervalRight::FromLevel >::value and
                level_geq< typename TIntervalRight::ToLevel, typename TIntervalLeft::ToLevel >::value),
            "check that the intervals are provided in order and are not subsets of each other");

        GRIDTOOLS_STATIC_ASSERT(
            (level_leq< typename TIntervalLeft::FromLevel, typename TIntervalRight::FromLevel >::value and
                level_geq< typename TIntervalRight::ToLevel, typename TIntervalLeft::ToLevel >::value),
            "check that the intervals are provided in order and are not subsets of each other");

        GRIDTOOLS_STATIC_ASSERT((join_interval_is_contiguous< TIntervalLeft, TIntervalRight >::value),
            "check that the intervals are contiguous");

        typedef interval< typename TIntervalLeft::FromLevel, typename TIntervalRight::ToLevel > type;
    };

    template < typename TIntervalLeft, typename TIntervalRight >
    using join_interval_t = typename join_interval< TIntervalLeft, TIntervalRight >::type;

    template < typename... TIntervals >
    struct make_axis {
        using levels = boost::mpl::vector< typename TIntervals::FromLevel..., typename TIntervals::ToLevel... >;

        using sorted_levels = typename boost::mpl::sort< levels, level_leq< boost::mpl::_, boost::mpl::_ > >::type;

        using smallest_level = typename boost::mpl::front< sorted_levels >::type;
        using biggest_level = typename boost::mpl::back< sorted_levels >::type;

        static const int left_index = level_to_index< smallest_level >::value - 1;
        static const int right_index = level_to_index< biggest_level >::value + 1;

        typedef typename index_to_level< static_int< left_index > >::type left_axis_level;
        typedef typename index_to_level< static_int< right_index > >::type right_axis_level;

        // user protection: make_axis will not add an additional splitter
        GRIDTOOLS_STATIC_ASSERT((left_axis_level::Splitter::value == smallest_level::Splitter::value),
            "the lowest level must not start at the smallest possible offset (solution: add an additional splitter)");
        GRIDTOOLS_STATIC_ASSERT((right_axis_level::Splitter::value == biggest_level::Splitter::value),
            "the highest level must not start at the biggest possible offset (solution: add an additional splitter)");

        typedef interval< left_axis_level, right_axis_level > type;
    };

    template < typename... TIntervals >
    using make_axis_t = typename make_axis< TIntervals... >::type;
#endif
} // namespace gridtools

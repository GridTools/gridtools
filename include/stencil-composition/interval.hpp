#pragma once

#include <boost/static_assert.hpp>
#include <boost/mpl/integral_c.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/range_c.hpp>
#include "level.hpp"
#include "../common/host_device.hpp"

namespace gridtools {
    /**
     * @struct Interval
     * Structure defining a closed interval on an axis given two levels
     */
    template<
        typename TFromLevel,
        typename TToLevel>
    struct interval
    {
        // HACK allow implicit conversion from the from level to any interval starting with the from level
        // (due to this trick we can search all do method overloads starting at a given from position)
        GT_FUNCTION
        interval() {};

        GT_FUNCTION
        interval(TFromLevel) {};

      static void static_info(){
          printf("level \"from\": splitter %d, offset %d \n", TFromLevel::Splitter::value, TFromLevel::Offset::value);
          printf("level \"to\": splitter %d, offset %d \n", TToLevel::Splitter::value, TToLevel::Offset::value);
      }
        // check the parameters are of type level
        GRIDTOOLS_STATIC_ASSERT(is_level<TFromLevel>::value, "check the first template parameter is of type level");
        GRIDTOOLS_STATIC_ASSERT(is_level<TToLevel>::value, "check the second template parameter is of type level");

        // check the from level is lower or equal to the to level
        GRIDTOOLS_STATIC_ASSERT(
                            (TFromLevel::Splitter::value < TToLevel::Splitter::value) ||
                            (TFromLevel::Splitter::value == TToLevel::Splitter::value && TFromLevel::Offset::value <= TToLevel::Offset::value),
                            "check the from level is lower or equal to the to level"
            );

        // define the from and to splitter indexes
        typedef TFromLevel FromLevel;
        typedef TToLevel ToLevel;
    };

    /**
     * @struct is_interval
     * Trait returning true it the template parameter is an interval
     */
    template<typename T>
    struct is_interval : boost::mpl::false_ {};

    template<
        typename TFromLevel,
        typename TToLevel>
    struct is_interval<interval<TFromLevel, TToLevel> > : boost::mpl::true_ {};

    /**
     * @struct interval_from_index
     * Meta function returning the interval from level index
     */
    template<typename TInterval>
    struct interval_from_index;

    template<
        typename TFromLevel,
        typename TToLevel>
    struct interval_from_index<interval<TFromLevel, TToLevel> > : level_to_index<TFromLevel> {};

    /**
     * @struct interval_to_index
     * Meta function returning the interval to level index
     */
    template<typename TInterval>
    struct interval_to_index;

    template<
        typename TFromLevel,
        typename TToLevel>
    struct interval_to_index<interval<TFromLevel, TToLevel> > : level_to_index<TToLevel> {};

    /**
     * @struct make_interval
     * Meta function computing an interval given a from and a to level index
     */
    template<
        typename TFromIndex,
        typename ToIndex>
    struct make_interval
    {
        typedef interval<
            typename index_to_level<TFromIndex>::type,
            typename index_to_level<ToIndex>::type
            > type;
    };
} // namespace gridtools

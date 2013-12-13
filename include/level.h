#pragma once

#include <iostream>

#include <boost/config.hpp>
#include <boost/static_assert.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/integral_c.hpp>
#include <boost/mpl/bool.hpp>

namespace gridtools {
    // define the level offset limit
    const int cLevelOffsetLimit = 3;

    /**
     * @struct Level
     * Structure defining an axis position relative to a splitter
     */
    template<
        int VSplitter, 
        int VOffset>
    struct level 
    {
        // check offset and splitter value ranges
        // (note that non negative splitter values simplify the index computation)
        BOOST_STATIC_ASSERT(VSplitter >= 0 && VOffset != 0);
        BOOST_STATIC_ASSERT(-cLevelOffsetLimit <= VOffset && VOffset <= cLevelOffsetLimit);
    
        // define splitter and level offset
        typedef boost::mpl::integral_c<int, VSplitter> Splitter;
        typedef boost::mpl::integral_c<int, VOffset> Offset;
    };

    /**
     * @struct is_level
     * Trait returning true it the template parameter is a level
     */
    template<typename T>
    struct is_level : boost::mpl::false_ {};

    template<
        int VSplitter, 
        int VOffset>
    struct is_level<level<VSplitter, VOffset> > : boost::mpl::true_ {};

    /**
     * @struct level_to_index
     * Meta function computing a unique index given a level
     */
    template<typename TLevel>
    struct level_to_index
    {
        // extract offset and splitter
        typedef typename TLevel::Splitter Splitter;
        typedef typename TLevel::Offset Offset;
    
        // define the splitter and offset indexes
        typedef boost::mpl::integral_c<int, 2 * cLevelOffsetLimit * Splitter::value> SplitterIndex;
        typedef boost::mpl::integral_c<int, (Offset::value < 0 ? Offset::value : Offset::value - 1) + cLevelOffsetLimit> OffsetIndex;
    
        // define the index value
        BOOST_STATIC_CONSTANT(int, value = SplitterIndex::value + OffsetIndex::value);
        typedef boost::mpl::integral_c<int, value> type;
    };

    /**
     * @struct index_to_level
     * Meta function converting a unique index back into a level
     */
    template<typename TIndex>
    struct index_to_level
    {
        // define splitter and offset values
        typedef boost::mpl::integral_c<int, TIndex::value / (2 * cLevelOffsetLimit)> Splitter;
        typedef boost::mpl::integral_c<int, TIndex::value % (2 * cLevelOffsetLimit) - cLevelOffsetLimit> OffsetIndex;
        typedef boost::mpl::integral_c<int, OffsetIndex::value < 0 ? OffsetIndex::value : OffsetIndex::value + 1> Offset;

        // define the level
        typedef level<Splitter::value, Offset::value> type;
    };

    /**
     * @struct make_range
     * Meta function converting two level indexes into a range
     */
    template<
        typename TFromIndex,
        typename TToIndex>
    struct make_range
    {
        typedef boost::mpl::range_c<int, TFromIndex::value, TToIndex::value + 1> type;
    };

    template <int F, int T>
    std::ostream& operator<<(std::ostream & s, level<F,T> const &) {
        return s << "(" << level<F,T>::Splitter::value << ", "
                 << level<F,T>::Offset::value << ")";
    }
} // namespace gridtools

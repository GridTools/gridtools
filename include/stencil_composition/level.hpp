/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#pragma once

#include <boost/config.hpp>
#include <boost/static_assert.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/integral_c.hpp>
#include <boost/mpl/bool.hpp>
#include "common/defs.hpp"

namespace gridtools {
    // define the level offset limit
    const short_t cLevelOffsetLimit = 3;
    /**
     * @struct Level
     * Structure defining an axis position relative to a splitter
     */
    template < uint_t VSplitter, int_t VOffset >
    struct level {
        // check offset and splitter value ranges
        // (note that non negative splitter values simplify the index computation)
        GRIDTOOLS_STATIC_ASSERT(VSplitter >= 0 && VOffset != 0, "check offset and splitter value ranges \n\
         (note that non negative splitter values simplify the index computation)");
        GRIDTOOLS_STATIC_ASSERT(
            -cLevelOffsetLimit <= VOffset && VOffset <= cLevelOffsetLimit, "check offset and splitter value ranges \n\
         (note that non negative splitter values simplify the index computation)");

#ifdef CXX11_ENABLED
        // define splitter and level offset
        typedef static_uint< VSplitter > Splitter;
        typedef static_int< VOffset > Offset;
#else
        typedef boost::mpl::integral_c< uint_t, VSplitter > Splitter;
        typedef boost::mpl::integral_c< int_t, VOffset > Offset;
#endif
    };

    /**
     * @struct is_level
     * Trait returning true it the template parameter is a level
     */
    template < typename T >
    struct is_level : boost::mpl::false_ {};

    template < uint_t VSplitter, int_t VOffset >
    struct is_level< level< VSplitter, VOffset > > : boost::mpl::true_ {};

    /**
     * @struct level_to_index
     * Meta function computing a unique index given a level
     */
    template < typename TLevel >
    struct level_to_index {
        // extract offset and splitter
        typedef typename TLevel::Splitter Splitter;
        typedef typename TLevel::Offset Offset;

        typedef static_uint< 2 * cLevelOffsetLimit * Splitter::value > SplitterIndex;
        typedef static_int< (Offset::value < 0 ? Offset::value : Offset::value - 1) + cLevelOffsetLimit > OffsetIndex;

        // define the index value
        BOOST_STATIC_CONSTANT(int_t, value = SplitterIndex::value + OffsetIndex::value);
        typedef static_int< value > type;
    };

    /**
     * @struct index_to_level
     * Meta function converting a unique index back into a level
     */
    template < typename TIndex >
    struct index_to_level {
        // define splitter and offset values
        typedef static_uint< TIndex::value / (2 * cLevelOffsetLimit) > Splitter;
        typedef static_int< TIndex::value % (2 * cLevelOffsetLimit) - cLevelOffsetLimit > OffsetIndex;
        typedef static_int < OffsetIndex::value< 0 ? OffsetIndex::value : OffsetIndex::value + 1 > Offset;

        // define the level
        typedef level< Splitter::value, Offset::value > type;
    };

    /**
     * @struct make_range
     * Meta function converting two level indexes into a range
     */
    template < typename TFromIndex, typename TToIndex >
    struct make_range {
        typedef boost::mpl::range_c< int_t, TFromIndex::value, TToIndex::value + 1 > type;
    };

    template < uint_t F, int_t T >
    std::ostream &operator<<(std::ostream &s, level< F, T > const &) {
        return s << "(" << level< F, T >::Splitter::value << ", " << level< F, T >::Offset::value << ")";
    }
} // namespace gridtools

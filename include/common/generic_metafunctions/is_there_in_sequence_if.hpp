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

#include <boost/mpl/is_sequence.hpp>
#include <boost/mpl/find.hpp>
#include "is_not_same.hpp"

namespace gridtools {

    /**
     * @struct is_there_in_sequence_if
     * return true if the predicate returns true when applied, for at least one of the elements in the Sequence
     */
    template < typename Sequence, typename Pred >
    struct is_there_in_sequence_if {
        GRIDTOOLS_STATIC_ASSERT((boost::mpl::is_sequence< Sequence >::value), "Wrong input sequence");
        typedef typename is_not_same< typename boost::mpl::find_if< Sequence, Pred >::type,
            typename boost::mpl::end< Sequence >::type >::type type;
        BOOST_STATIC_CONSTANT(bool, value = (type::value));
    };
}

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

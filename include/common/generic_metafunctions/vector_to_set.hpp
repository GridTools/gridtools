#pragma once
#include <boost/mpl/set/set0.hpp>
#include <boost/mpl/insert.hpp>
#include <boost/mpl/fold.hpp>

namespace gridtools {

    /**
     * @struct vector_to_map
     * convert a vector sequence into a set sequence
     */
    template < typename Vec >
    struct vector_to_set {
        typedef typename boost::mpl::fold< Vec,
            boost::mpl::set0<>,
            boost::mpl::insert< boost::mpl::_1, boost::mpl::_2 > >::type type;
    };

} // namespace gridtools

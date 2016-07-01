#pragma once
#include <boost/mpl/map.hpp>
#include <boost/fusion/mpl/insert.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/fusion/algorithm/transformation/insert.hpp>
#include <boost/fusion/include/insert.hpp>
#include <boost/fusion/algorithm/transformation/push_back.hpp>
#include <boost/fusion/include/push_back.hpp>

namespace gridtools {

    /**
     * @struct vector_to_map
     * convert a vector of pairs into a make_pair
     */
    template < typename Vec >
    struct vector_to_map {
        typedef typename boost::mpl::fold< Vec,
            boost::mpl::map0<>,
            boost::mpl::insert< boost::mpl::_1, boost::mpl::_2 > >::type type;
    };

} // namespace gridtools

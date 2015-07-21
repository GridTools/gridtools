/*
 * vector_to_map.hpp
 *
 *  Created on: Jul 21, 2015
 *      Author: cosuna
 */

#pragma once
#include <boost/mpl/map.hpp>
#include <boost/mpl/insert.hpp>
#include <boost/mpl/fold.hpp>

namespace gridtools {

template<typename Vec>
struct vector_to_map
{
    typedef typename boost::mpl::fold<
        Vec,
        boost::mpl::map0<>,
        boost::mpl::insert< boost::mpl::_1, boost::mpl::_2>
    >::type type;
};

} //namespace gridtools

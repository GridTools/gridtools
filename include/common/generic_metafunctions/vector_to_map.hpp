/*
 * vector_to_map.hpp
 *
 *  Created on: Jul 21, 2015
 *      Author: cosuna
 */

#pragma once
#include <boost/mpl/map.hpp>
#include <boost/fusion/mpl/insert.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/fusion/algorithm/transformation/insert.hpp>
#include <boost/fusion/include/insert.hpp>
#include <boost/fusion/algorithm/transformation/push_back.hpp>
#include <boost/fusion/include/push_back.hpp>

namespace gridtools {

template<typename Vec>
struct vector_to_map
{
    typedef typename boost::mpl::fold<
        Vec,
        boost::mpl::map0<>,
        boost::mpl::insert< boost::mpl::_1, boost::mpl::_2 >
    >::type type;
};

template<typename Vec>
struct mpl_vector_to_fusion_map
{
    typedef typename boost::mpl::fold<
        Vec,
        boost::fusion::map<>,
        boost::fusion::result_of::push_back< boost::mpl::_1, boost::mpl::_2 >
    >::type type;
};

} //namespace gridtools

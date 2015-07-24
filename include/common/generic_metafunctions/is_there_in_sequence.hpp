/*
 * is_there_in_sequence.hpp
 *
 *  Created on: Jul 17, 2015
 *      Author: cosuna
 */

#pragma once

#include <boost/mpl/is_sequence.hpp>
#include <boost/mpl/find.hpp>
#include "is_not_same.hpp"

namespace gridtools {

/**
 * @struct is_there_in_sequence
 * returns true if the Key is found in the sequence
 */
template<typename Sequence, typename Key>
struct is_there_in_sequence
{
    GRIDTOOLS_STATIC_ASSERT((boost::mpl::is_sequence<Sequence>::value), "Wrong input sequence");
    typedef typename is_not_same<
        typename boost::mpl::find<Sequence, Key>::type,
        typename boost::mpl::end<Sequence>::type
    >::type type;
    BOOST_STATIC_CONSTANT(bool, value = (type::value));
};

}

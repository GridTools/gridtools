/*
 * test_is_there_in_sequence.cpp
 *
 *  Created on: Jul 17, 2015
 *      Author: cosuna
 */

#include "gtest/gtest.h"
#include "defs.hpp"
#include <common/generic_metafunctions/is_there_in_sequence.hpp>
#include <common/generic_metafunctions/is_there_in_sequence_if.hpp>

using namespace gridtools;

TEST(is_there_in_sequence, is_there_in_sequence)
{
    typedef boost::mpl::vector<int, float, char> seq_t;

    GRIDTOOLS_STATIC_ASSERT((is_there_in_sequence<seq_t, int>::value),"ERROR");
    GRIDTOOLS_STATIC_ASSERT((is_there_in_sequence<seq_t, char>::value),"ERROR");
    GRIDTOOLS_STATIC_ASSERT((! is_there_in_sequence<seq_t, long>::value),"ERROR");

    ASSERT_TRUE(true);
}

TEST(is_there_in_sequence, is_there_in_sequence_if)
{
    typedef boost::mpl::vector<int, float, char> seq_t;

    GRIDTOOLS_STATIC_ASSERT((is_there_in_sequence_if<seq_t, boost::is_same<boost::mpl::_, char> >::value),"ERROR");
    GRIDTOOLS_STATIC_ASSERT((is_there_in_sequence_if<seq_t, boost::is_same<boost::mpl::_, int> >::value),"ERROR");
    GRIDTOOLS_STATIC_ASSERT((! is_there_in_sequence_if<seq_t, boost::is_same<boost::mpl::_, long> >::value),"ERROR");

    ASSERT_TRUE(true);
}


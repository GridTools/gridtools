#include "gtest/gtest.h"
#include "defs.hpp"
#include "common/generic_metafunctions/is_there_in_sequence_if.hpp"

using namespace gridtools;

TEST(is_there_in_sequence_if, is_there_in_sequence_if)
{
    typedef boost::mpl::vector<int, float, char> seq_t;

    GRIDTOOLS_STATIC_ASSERT((is_there_in_sequence_if<seq_t, boost::is_same<boost::mpl::_, char> >::value),"ERROR");
    GRIDTOOLS_STATIC_ASSERT((is_there_in_sequence_if<seq_t, boost::is_same<boost::mpl::_, int> >::value),"ERROR");
    GRIDTOOLS_STATIC_ASSERT((! is_there_in_sequence_if<seq_t, boost::is_same<boost::mpl::_, long> >::value),"ERROR");

    ASSERT_TRUE(true);
}


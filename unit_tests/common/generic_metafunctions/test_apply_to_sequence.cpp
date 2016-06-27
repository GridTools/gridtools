#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include "common/defs.hpp"
#include "common/generic_metafunctions/apply_to_sequence.hpp"

using namespace gridtools;

template<int V>
struct metadata{
    static const int val=V;
};

template<typename Metadata>
struct extract_metadata
{
    typedef static_int<Metadata::val> type;
};

TEST(test_apply_to_sequence, test)
{
    typedef boost::mpl::vector3<metadata<3>, metadata<5>, metadata<8> > seq;

    typedef apply_to_sequence<seq, extract_metadata>::type res;
    GRIDTOOLS_STATIC_ASSERT((boost::mpl::equal<res, boost::mpl::vector<static_int<3>, static_int<5>, static_int<8> > >::value), "Wrong RESULT");

    ASSERT_TRUE(true);
}


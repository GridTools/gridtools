#include "gtest/gtest.h"

#define SILENT_RUN
#include <boost/mpl/map/map10.hpp>
#include "stencil-composition/accessor_metafunctions.h"
#include "stencil-composition/iterate_domain_evaluator.h"

TEST(accessor, copy_const) {

    using namespace gridtools;

    typedef accessor<0, range<-1,0,0,0>, 3> accessor_t;
    accessor<0, range<-1,0,0,0>, 3> in(1,2,3);
    accessor<1, range<-1,0,0,0>, 3> out(in);

    ASSERT_TRUE(in.get<0>() == 3 && out.get<0>()==3);
    ASSERT_TRUE(in.get<1>() == 2 && out.get<1>()==2);
    ASSERT_TRUE(in.get<2>() == 1 && out.get<2>()==1);

    typedef boost::mpl::map1<
        boost::mpl::pair<
            boost::mpl::integral_c<int, 0>, boost::mpl::integral_c<int, 8>
        >
    > ArgsMap;

    typedef remap_arg_type<accessor_t, ArgsMap>::type remap_arg_t;

    BOOST_STATIC_ASSERT((is_accessor<remap_arg_t>::value));
    BOOST_STATIC_ASSERT((accessor_index<remap_arg_t>::value == 8));

    ASSERT_TRUE(remap_arg_t(in).get<0>() == 3);
    ASSERT_TRUE(remap_arg_t(in).get<1>() == 2);
    ASSERT_TRUE(remap_arg_t(in).get<2>() == 1);
}


int main(int argc, char** argv)
{

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

#include "gtest/gtest.h"

// #include <boost/mpl/map/map10.hpp>
// #include "stencil-composition/accessor_metafunctions.hpp"
// #include "stencil-composition/iterate_domain_evaluator.hpp"

TEST(accessor, copy_const) {

    // using namespace gridtools;
//TODOCOSUNA not working due to problems with the copy ctor of the accessors

//    typedef accessor<0, range<-1,0,0,0>, 3> accessor_t;
//    accessor<0, range<-1,0,0,0>, 3> in(1,2,3);
//    accessor<1, range<-1,0,0,0>, 3> out(in);
//
//    ASSERT_TRUE(in.get<0>() == 3 && out.get<0>()==3);
//    ASSERT_TRUE(in.get<1>() == 2 && out.get<1>()==2);
//    ASSERT_TRUE(in.get<2>() == 1 && out.get<2>()==1);
//
//    typedef boost::mpl::map1<
//        boost::mpl::pair<
//            boost::mpl::integral_c<int, 0>, boost::mpl::integral_c<int, 8>
//        >
//    > ArgsMap;
//
//    typedef remap_accessor_type<accessor_t, ArgsMap>::type remap_accessor_t;
//
//    BOOST_STATIC_ASSERT((is_accessor<remap_accessor_t>::value));
//    BOOST_STATIC_ASSERT((accessor_index<remap_accessor_t>::value == 8));
//
//    ASSERT_TRUE(remap_accessor_t(in).get<0>() == 3);
//    ASSERT_TRUE(remap_accessor_t(in).get<1>() == 2);
//    ASSERT_TRUE(remap_accessor_t(in).get<2>() == 1);
}

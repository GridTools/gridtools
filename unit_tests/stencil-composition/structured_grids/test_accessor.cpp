#include "gtest/gtest.h"

// #include <boost/mpl/map/map10.hpp>
#include <common/defs.hpp>
#include <stencil-composition/structured_grids/accessor.hpp>
#include <stencil-composition/structured_grids/accessor_metafunctions.hpp>
// #include "stencil-composition/iterate_domain_remapper.hpp"

TEST(accessor, is_accessor) {
    using namespace gridtools;
    GRIDTOOLS_STATIC_ASSERT((is_accessor<accessor<6, enumtype::inout, extent<3,4,4,5> > >::value) == true, "");
    GRIDTOOLS_STATIC_ASSERT((is_accessor<accessor<-2,  enumtype::in> >::value) == true, "");
    GRIDTOOLS_STATIC_ASSERT((is_accessor<int>::value) == false, "");
    GRIDTOOLS_STATIC_ASSERT((is_accessor<double&>::value) == false, "");
    GRIDTOOLS_STATIC_ASSERT((is_accessor<double const&>::value) == false, "");
}

TEST(accessor, copy_const) {

    // using namespace gridtools;
//TODOCOSUNA not working due to problems with the copy ctor of the accessors

//    typedef accessor<0, extent<-1,0,0,0>, 3> accessor_t;
//    accessor<0, extent<-1,0,0,0>, 3> in(1,2,3);
//    accessor<1, extent<-1,0,0,0>, 3> out(in);
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

#include "gtest/gtest.h"
#include <stencil-composition/make_computation.hpp>

TEST(storage_alignment, test_aligned) {
    using namespace gridtools;
    using namespace enumtype;

    typedef backend<Host, Block>::storage_info<0, layout_map<2,1,0>, halo<1,2,3> > meta_t;
    meta_t m1(1,2,3);
    ASSERT_TRUE((m1.dims<0>()==2));
    ASSERT_TRUE((m1.dims<1>()==4));
    ASSERT_TRUE((m1.dims<2>()==6));

}

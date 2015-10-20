#include "gtest/gtest.h"
#include <stencil-composition/make_computation.hpp>

TEST(storage_alignment, test_aligned) {
    using namespace gridtools;
    using namespace enumtype;

    typedef gridtools::backend<Cuda, Block>::storage_info<0, gridtools::layout_map<2,1,0>, gridtools::padding<1,2,3> > meta_gpu_t;
    meta_gpu_t m2(1,32,63);
    ASSERT_TRUE((m2.dims<0>()==32+1));
    ASSERT_TRUE((m2.dims<1>()==64+2));
    ASSERT_TRUE((m2.dims<2>()==64+3));

}

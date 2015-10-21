#include "gtest/gtest.h"
#include <stencil-composition/make_computation.hpp>

TEST(storage_alignment, test_aligned) {
    using namespace gridtools;
    using namespace enumtype;

    typedef gridtools::backend<Cuda, Block>::storage_info<0, gridtools::layout_map<2,1,0>, gridtools::padding<1,2,3> > meta_gpu_t;
    meta_gpu_t m2(1,32,63);
    ASSERT_TRUE((m2.dims<0>()==64));
    ASSERT_TRUE((m2.dims<1>()==64));
    ASSERT_TRUE((m2.dims<2>()==96));

    typedef gridtools::backend<Cuda, Block>::temporary_storage_type< int, meta_gpu_t>::type tmp_storage_t;

    typedef meta_storage_tmp< typename tmp_storage_t::type::basic_type::meta_data_t, tile<32, 1, 1>, tile<32, 1, 1> > tmp_meta_gpu_t;

    tmp_meta_gpu_t m_block(0,0,15,1,1);
    ASSERT_TRUE((m_block.dims<0>()==96));//3 blocks wide, (since halos in both directions 32-1->32+1+1)
    ASSERT_TRUE((m_block.dims<1>()==96));//3 blocks wide (since halos in both directions 32-1->32+1+2)
    ASSERT_TRUE((m_block.dims<2>()==64));//2 blocks wide (since padding 0->32+3)

}

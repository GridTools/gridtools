#include "gtest/gtest.h"
#include <stencil-composition/make_computation.hpp>

TEST(storage_alignment, test_aligned) {
    using namespace gridtools;
    using namespace enumtype;

    //define three storage types, with different layouts
    typedef gridtools::backend<Cuda, Block>::storage_info<0, gridtools::layout_map<2,1,0>, gridtools::halo<1,2,3> > meta_gpu1_t;
    typedef gridtools::backend<Cuda, Block>::storage_info<0, gridtools::layout_map<0,2,1>, gridtools::halo<1,2,3> > meta_gpu2_t;
    typedef gridtools::backend<Cuda, Block>::storage_info<0, gridtools::layout_map<1,0,2>, gridtools::halo<1,2,3> > meta_gpu3_t;

    meta_gpu1_t m1(1,32,63);

    meta_gpu2_t m2(1,32,63);

    meta_gpu3_t m3(1,32,63);

    //check that the dimension with stride 1 is aligned
    ASSERT_TRUE((m1.dims<0>()==32));
    ASSERT_TRUE((m2.dims<1>()==64));
    ASSERT_TRUE((m3.dims<2>()==96));

    //define three temporary storage types, with different layouts
    typedef gridtools::backend<Cuda, Block>::temporary_storage_type< int, meta_gpu1_t>::type tmp_storage1_t;
    typedef gridtools::backend<Cuda, Block>::temporary_storage_type< int, meta_gpu2_t>::type tmp_storage2_t;
    typedef gridtools::backend<Cuda, Block>::temporary_storage_type< int, meta_gpu3_t>::type tmp_storage3_t;

    typedef meta_storage_tmp< typename tmp_storage1_t::type::basic_type::meta_data_t, tile<32, 1, 1>, tile<32, 1, 1> > tmp_meta_gpu1_t;
    typedef meta_storage_tmp< typename tmp_storage2_t::type::basic_type::meta_data_t, tile<32, 1, 1>, tile<32, 1, 1> > tmp_meta_gpu2_t;
    typedef meta_storage_tmp< typename tmp_storage3_t::type::basic_type::meta_data_t, tile<32, 1, 1>, tile<32, 1, 1> > tmp_meta_gpu3_t;

    tmp_meta_gpu1_t m_block1(0,0,15,1,1);
    tmp_meta_gpu2_t m_block2(0,0,15,1,1);
    tmp_meta_gpu3_t m_block3(0,0,15,1,1);

    //check that the dimension with stride 1 is aligned
    ASSERT_TRUE((m_block1.dims<0>()==64));//2 blocks wide
    ASSERT_TRUE((m_block2.dims<1>()==64));//2 blocks wide
    ASSERT_TRUE((m_block3.dims<2>()==32));//1 block wide

}

#include "gtest/gtest.h"
#include <stencil-composition/stencil-composition.hpp>
#include <storage/storage.hpp>

using namespace gridtools;

TEST(storage_info, test_interface) {
#if defined(CXX11_ENABLED) && defined(NDEBUG)
    //unaligned meta_storage test cases
    typedef gridtools::layout_map<0,1,2,3> layout_t;
    constexpr gridtools::meta_storage_base<0,layout_t,false> meta_{11, 12, 13, 14};
    GRIDTOOLS_STATIC_ASSERT((meta_.dim<0>()==11), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.dim<1>()==12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.dim<2>()==13), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.dim<3>()==14), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_.strides(3)==14), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides(2)==14*13), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides(1)==14*13*12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides(0)==14*13*12*11), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_.strides<3>()==1), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides<2>()==14), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides<1>()==14*13), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides<0>()==14*13*12), "error");

    //aligned meta_storage test cases
    using halo_t = gridtools::halo<0,0,0>;
    using align_t = gridtools::aligned<32>;
    constexpr gridtools::meta_storage_aligned<gridtools::meta_storage_base<0,gridtools::layout_map<0,1,2>,false>, align_t, halo_t> meta_aligned_1 {11, 12, 13};
    constexpr gridtools::meta_storage_aligned<gridtools::meta_storage_base<0,gridtools::layout_map<0,2,1>,false>, align_t, halo_t> meta_aligned_2 {11, 12, 13};
    constexpr gridtools::meta_storage_aligned<gridtools::meta_storage_base<0,gridtools::layout_map<2,1,0>,false>, align_t, halo_t> meta_aligned_3 {11, 12, 13};

    //check unaligned dimensions with either templated method or method that takes an argument
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.unaligned_dims<0>()==11), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.unaligned_dims<1>()==12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.unaligned_dims<2>()==13), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.unaligned_dims(0)==11), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.unaligned_dims(1)==12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.unaligned_dims(2)==13), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.unaligned_dims<0>()==11), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.unaligned_dims<1>()==12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.unaligned_dims<2>()==13), "error");

    //check aligned dimensions with either templated method or method that takes an argument
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.dim<0>()==11), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.dim<1>()==12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.dim<2>()==32), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.dim(0)==11), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.dim(1)==32), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.dim(2)==13), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.dim<0>()==32), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.dim<1>()==12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.dim<2>()==13), "error");


    //check unaligned strides with either templated method or method that takes an argument
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.unaligned_strides(2)==13), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.unaligned_strides(1)==13*12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.unaligned_strides(0)==13*12*11), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.unaligned_strides<2>()==12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.unaligned_strides<1>()==1), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.unaligned_strides<0>()==12*13), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.unaligned_strides(2)==11), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.unaligned_strides(1)==11*12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.unaligned_strides(0)==11*12*13), "error");

    //check unaligned strides with either templated method or method that takes an argument
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.strides(2)==32), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.strides(1)==32*12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.strides(0)==32*12*11), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.strides<2>()==32), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.strides<1>()==1), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.strides<0>()==32*13), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.strides(2)==32), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.strides(1)==32*12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.strides(0)==32*12*13), "error");

#else
    typedef gridtools::layout_map< 0, 1, 2 > layout_t;
    gridtools::meta_storage_base< 0, layout_t, false > meta_(11, 12, 13);
    ASSERT_TRUE((meta_.dim< 0 >() == 11));
    ASSERT_TRUE((meta_.dim< 1 >() == 12));
    ASSERT_TRUE((meta_.dim< 2 >() == 13));

    ASSERT_TRUE((meta_.strides(2) == 13));
    ASSERT_TRUE((meta_.strides(1) == 13 * 12));
    ASSERT_TRUE((meta_.strides(0) == 13 * 12 * 11));

    ASSERT_TRUE((meta_.strides< 2 >(meta_.strides()) == 1));
    ASSERT_TRUE((meta_.strides< 1 >(meta_.strides()) == 13));
    ASSERT_TRUE((meta_.strides< 0 >(meta_.strides()) == 13 * 12));
#endif
}

#ifdef CXX11_ENABLED
TEST(storage_info, meta_storage_extender) {

    GRIDTOOLS_STATIC_ASSERT((boost::is_same< meta_storage_extender_impl< layout_map< 0, 1, 2, 3 >, 1 >::type,
                                layout_map< 1, 2, 3, 4, 0 > >::value),
        "Error");

    GRIDTOOLS_STATIC_ASSERT((boost::is_same< meta_storage_extender_impl< layout_map< 3, 1, 2, 0 >, 1 >::type,
                                layout_map< 4, 2, 3, 1, 0 > >::value),
        "Error");

    typedef meta_storage_base< 0, layout_map< 0, 1, 2, 3 >, false > meta_storage1_t;
    typedef meta_storage_base< 0, layout_map< 1, 2, 3, 4, 0 >, false > meta_storage1_ext_t;

    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< meta_storage_extender_impl< meta_storage1_t, 1 >::type, meta_storage1_ext_t >::value),
        "Error");

    typedef meta_storage_base< 0, layout_map< 3, 4, 5, 6, 0, 1, 2 >, false > meta_storage_multi_ext_t;

    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< meta_storage_extender_impl< meta_storage1_t, 3 >::type, meta_storage_multi_ext_t >::value),
        "Error");

    typedef meta_storage_aligned< meta_storage1_t, aligned< 0 >, halo< 0, 0, 0, 0 > > meta_storage_aligned_t;
    typedef meta_storage_aligned< meta_storage1_ext_t, aligned< 0 >, halo< 0, 0, 0, 0, 0 > > meta_storage_aligned_ext_t;

    typedef meta_storage< meta_storage_aligned_t > meta_storage2_t;

    typedef meta_storage< meta_storage_aligned_ext_t > meta_storage2_ext_t;

    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< meta_storage_extender_impl< meta_storage2_t, 1 >::type, meta_storage2_ext_t >::value),
        "Error");

    meta_storage_aligned_t meta(34, 23, 54, 5);
    auto extended_meta = meta_storage_extender()(meta, 10);

    ASSERT_TRUE((extended_meta.template dim< 0 >() == 34));
    ASSERT_TRUE((extended_meta.template dim< 1 >() == 23));
    ASSERT_TRUE((extended_meta.template dim< 2 >() == 54));
    ASSERT_TRUE((extended_meta.template dim< 3 >() == 5));
    ASSERT_TRUE((extended_meta.template dim< 4 >() == 10));
}

#endif

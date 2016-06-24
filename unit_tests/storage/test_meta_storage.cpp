#include "gtest/gtest.h"
#include <storage/storage-facility.hpp>

using namespace gridtools;

TEST(storage_info, test_component) {
    typedef layout_map< 0, 1, 2 > layout;
#ifdef CXX11_ENABLED
    typedef storage_traits< enumtype::Host >::meta_storage_type< 0, layout > meta_data_t;
    typedef storage_traits< enumtype::Host >::storage_type< float, meta_data_t > storage_t;
#else
    typedef storage_traits< enumtype::Host >::meta_storage_type< 0, layout >::type meta_data_t;
    typedef storage_traits< enumtype::Host >::storage_type< float, meta_data_t >::type storage_t;
#endif
    meta_data_t meta_obj(10, 10, 10);
    storage_t st_obj(meta_obj, "in");
}

TEST(storage_info, test_equality) {
    typedef gridtools::layout_map< 0, 1, 2 > layout_t1;
    typedef gridtools::meta_storage_base< 0, layout_t1, false > meta_t1;
    meta_t1 m0(11, 12, 13);
    meta_t1 m1(11, 12, 13);
    meta_t1 m2(12, 123, 13);
    ASSERT_TRUE((m0 == m1) && "storage info equality test failed!");
    ASSERT_TRUE((m1 == m0) && "storage info equality test failed!");
    ASSERT_TRUE(!(m2 == m0) && "storage info equality test failed!");
}

TEST(storage_info, test_interface) {
#if defined(CXX11_ENABLED) && defined(NDEBUG)
    // unaligned meta_storage test cases
    typedef gridtools::layout_map<0,1,2,3> layout_t;
    constexpr gridtools::meta_storage_base<0,layout_t,false> meta_{11, 12, 13, 14};
    GRIDTOOLS_STATIC_ASSERT((meta_.dim< 0 >() == 11), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.dim< 1 >() == 12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.dim< 2 >() == 13), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.dim< 3 >() == 14), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_.strides(3)==14), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides(2)==14*13), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides(1)==14*13*12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides(0)==14*13*12*11), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_.strides<3>()==1), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides<2>()==14), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides<1>()==14*13), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides<0>()==14*13*12), "error");

    // aligned meta_storage test cases
    using halo_t = gridtools::halo< 0, 0, 0 >;
    using align_t = gridtools::aligned< 32 >;
    constexpr gridtools::meta_storage_aligned<
        gridtools::meta_storage_base< 0, gridtools::layout_map< 0, 1, 2 >, false >,
        align_t,
        halo_t > meta_aligned_1{11, 12, 13};
    constexpr gridtools::meta_storage_aligned<
        gridtools::meta_storage_base< 0, gridtools::layout_map< 0, 2, 1 >, false >,
        align_t,
        halo_t > meta_aligned_2{11, 12, 13};
    constexpr gridtools::meta_storage_aligned<
        gridtools::meta_storage_base< 0, gridtools::layout_map< 2, 1, 0 >, false >,
        align_t,
        halo_t > meta_aligned_3{11, 12, 13};

    // check unaligned dimensions with either templated method or method that takes an argument
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.unaligned_dim< 0 >() == 11), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.unaligned_dim< 1 >() == 12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.unaligned_dim< 2 >() == 13), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.unaligned_dim(0) == 11), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.unaligned_dim(1) == 12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.unaligned_dim(2) == 13), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.unaligned_dim< 0 >() == 11), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.unaligned_dim< 1 >() == 12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.unaligned_dim< 2 >() == 13), "error");

    // check aligned dimensions with either templated method or method that takes an argument
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.dim< 0 >() == 11), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.dim< 1 >() == 12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.dim< 2 >() == 32), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.dim(0) == 11), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.dim(1) == 32), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.dim(2) == 13), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.dim< 0 >() == 32), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.dim< 1 >() == 12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.dim< 2 >() == 13), "error");

    // check unaligned strides with either templated method or method that takes an argument
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.unaligned_strides(2) == 13), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.unaligned_strides(1) == 13 * 12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.unaligned_strides(0) == 13 * 12 * 11), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.unaligned_strides< 2 >() == 12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.unaligned_strides< 1 >() == 1), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.unaligned_strides< 0 >() == 12 * 13), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.unaligned_strides(2) == 11), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.unaligned_strides(1) == 11 * 12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.unaligned_strides(0) == 11 * 12 * 13), "error");

    // check unaligned strides with either templated method or method that takes an argument
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.strides(2) == 32), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.strides(1) == 32 * 12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.strides(0) == 32 * 12 * 11), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.strides< 2 >() == 32), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.strides< 1 >() == 1), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.strides< 0 >() == 32 * 13), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.strides(2) == 32), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.strides(1) == 32 * 12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.strides(0) == 32 * 12 * 13), "error");
#else
    typedef gridtools::layout_map<0,1,2> layout_t;
    gridtools::meta_storage_base<0,layout_t,false> meta_(11, 12, 13);
    ASSERT_TRUE((meta_.dim< 0 >() == 11));
    ASSERT_TRUE((meta_.dim< 1 >() == 12));
    ASSERT_TRUE((meta_.dim< 2 >() == 13));

    ASSERT_TRUE((meta_.strides(2)==13));
    ASSERT_TRUE((meta_.strides(1)==13*12));
    ASSERT_TRUE((meta_.strides(0)==13*12*11));

    ASSERT_TRUE((meta_.strides<2>(meta_.strides())==1));
    ASSERT_TRUE((meta_.strides<1>(meta_.strides())==13));
    ASSERT_TRUE((meta_.strides<0>(meta_.strides())==13*12));

#ifdef CXX11_ENABLED // this checks are performed in cxx11 mode (without ndebug)
    // create simple aligned meta storage
    typedef gridtools::halo< 0, 0, 0 > halo_t1;
    typedef gridtools::aligned< 32 > align_t1;
    gridtools::meta_storage_aligned<
        gridtools::meta_storage_base< 0, gridtools::layout_map< 0, 1, 2 >, false >,
        align_t1,
        halo_t1 > meta_aligned_1nc(11, 12, 13);
    // check if unaligned dims and strides are correct
    ASSERT_TRUE((meta_aligned_1nc.unaligned_dim(0) == 11) && "error");
    ASSERT_TRUE((meta_aligned_1nc.unaligned_dim(1) == 12) && "error");
    ASSERT_TRUE((meta_aligned_1nc.unaligned_dim(2) == 13) && "error");
    ASSERT_TRUE((meta_aligned_1nc.unaligned_strides(2) == 13) && "error");
    ASSERT_TRUE((meta_aligned_1nc.unaligned_strides(1) == 13 * 12) && "error");
    ASSERT_TRUE((meta_aligned_1nc.unaligned_strides(0) == 13 * 12 * 11) && "error");
    //create a storage and pass meta_data
    gridtools::storage<gridtools::base_storage<gridtools::wrap_pointer<float>, decltype(meta_aligned_1nc), 1> > storage(meta_aligned_1nc, -1.0f);
    ASSERT_TRUE((storage.meta_data().unaligned_dim(0) == 11) && "error");
    ASSERT_TRUE((storage.meta_data().unaligned_dim(1) == 12) && "error");
    ASSERT_TRUE((storage.meta_data().unaligned_dim(2) == 13) && "error");
    ASSERT_TRUE((storage.meta_data().unaligned_strides(2) == 13) && "error");
    ASSERT_TRUE((storage.meta_data().unaligned_strides(1) == 13 * 12) && "error");
    ASSERT_TRUE((storage.meta_data().unaligned_strides(0) == 13 * 12 * 11) && "error");
#endif // CXX11_ENABLED

#endif // defined(CXX11_ENABLED) && defined(NDEBUG)
}

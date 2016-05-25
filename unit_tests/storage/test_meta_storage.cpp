#include "gtest/gtest.h"
#include <stencil-composition/stencil-composition.hpp>

using namespace gridtools;

TEST(storage_info, test_interface) {
#if defined(CXX11_ENABLED) && defined(NDEBUG)
    // unaligned meta_storage test cases
    typedef gridtools::layout_map<0,1,2,3> layout_t;
    constexpr gridtools::meta_storage_base<0,layout_t,false> meta_{11, 12, 13, 14};
    GRIDTOOLS_STATIC_ASSERT((meta_.dims<0>()==11), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.dims<1>()==12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.dims<2>()==13), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.dims<3>()==14), "error");

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
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.unaligned_dims< 0 >() == 11), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.unaligned_dims< 1 >() == 12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.unaligned_dims< 2 >() == 13), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.unaligned_dims(0) == 11), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.unaligned_dims(1) == 12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.unaligned_dims(2) == 13), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.unaligned_dims< 0 >() == 11), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.unaligned_dims< 1 >() == 12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.unaligned_dims< 2 >() == 13), "error");

    // check aligned dimensions with either templated method or method that takes an argument
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.dims< 0 >() == 11), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.dims< 1 >() == 12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_1.dims< 2 >() == 32), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.dims(0) == 11), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.dims(1) == 32), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_2.dims(2) == 13), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.dims< 0 >() == 32), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.dims< 1 >() == 12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_aligned_3.dims< 2 >() == 13), "error");

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
    ASSERT_TRUE((meta_.dims<0>()==11));
    ASSERT_TRUE((meta_.dims<1>()==12));
    ASSERT_TRUE((meta_.dims<2>()==13));

    ASSERT_TRUE((meta_.strides(2)==13));
    ASSERT_TRUE((meta_.strides(1)==13*12));
    ASSERT_TRUE((meta_.strides(0)==13*12*11));

    ASSERT_TRUE((meta_.strides<2>(meta_.strides())==1));
    ASSERT_TRUE((meta_.strides<1>(meta_.strides())==13));
    ASSERT_TRUE((meta_.strides<0>(meta_.strides())==13*12));
#endif
}

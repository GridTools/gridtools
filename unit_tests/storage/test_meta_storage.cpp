#include "gtest/gtest.h"
#include <stencil-composition/stencil-composition.hpp>

using namespace gridtools;

TEST(storage_info, test_interface) {
    typedef gridtools::layout_map<0,1,2,3,4> layout_t;
#if defined(CXX11_ENABLED) && defined(NDEBUG)
#ifndef __CUDACC__
    constexpr gridtools::meta_storage_base<static_int<0>,layout_t,false> meta_{11u, 12u, 13u, 14u, 15u};
#else
    constexpr gridtools::meta_storage_base<static_int<0>,layout_t,false> meta_{gridtools::static_uint<11>(), gridtools::static_uint<12>(), gridtools::static_uint<13>(), gridtools::static_uint<14>(), gridtools::static_uint<15>()};
#endif

    GRIDTOOLS_STATIC_ASSERT((meta_.dims<0>()==11), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.dims<1>()==12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.dims<2>()==13), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.dims<3>()==14), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_.strides(4)==15), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides(3)==15*14), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides(2)==15*14*13), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides(1)==15*14*13*12), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides(0)==15*14*13*12*11), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides<4>(meta_.strides())==1), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides<3>(meta_.strides())==15), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides<2>(meta_.strides())==15*14), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides<1>(meta_.strides())==15*14*13), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides<0>(meta_.strides())==15*14*13*12), "error");

    GRIDTOOLS_STATIC_ASSERT((meta_.strides<3>()==1), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides<2>()==14), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides<1>()==14*13), "error");
    GRIDTOOLS_STATIC_ASSERT((meta_.strides<0>()==14*13*12), "error");
#else // CXX11_ENABLED && defined(NDEBUG)

    gridtools::meta_storage_base<static_int<0>,layout_t,false> meta_(11u, 12u, 13u, 14u, 15u);

    ASSERT_TRUE((meta_.dims<0>()==(uint_t)11));
    ASSERT_TRUE((meta_.dims<1>()==(uint_t)12));
    ASSERT_TRUE((meta_.dims<2>()==(uint_t)13));

    ASSERT_TRUE((meta_.strides(4)==15));
    ASSERT_TRUE((meta_.strides(3)==15*14));
    ASSERT_TRUE((meta_.strides(2)==15*14*13));

    ASSERT_TRUE((meta_.strides<4>(meta_.strides())==1));
    ASSERT_TRUE((meta_.strides<3>(meta_.strides())==15));
    ASSERT_TRUE((meta_.strides<2>(meta_.strides())==15*14));
#endif
}

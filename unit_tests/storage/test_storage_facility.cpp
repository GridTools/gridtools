/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>

#include <type_traits>

#include <gridtools/common/gt_assert.hpp>
#include <gridtools/storage/storage_facility.hpp>
#include <gridtools/tools/backend_select.hpp>

using namespace gridtools;

// static type tests for x86 and naive backend
template <class Backend>
struct static_type_tests {
    using storage_traits_t = storage_traits<Backend>;

    /*########## STORAGE INFO CHECKS ########## */
    // storage info check
    typedef typename storage_traits_t::template storage_info_t<0, 3, halo<1, 2, 3>> storage_info_ty;
    static_assert(is_storage_info<storage_info_ty>::type::value, "");
    static_assert(
        std::is_same<storage_info_ty, storage_info<0, layout_map<0, 1, 2>, halo<1, 2, 3>, alignment<1>>>::value, "");

    // special layout
    typedef typename storage_traits_t::template special_storage_info_t<0, selector<1, 1, 0>, halo<1, 2, 3>>
        special_storage_info_ty;
    static_assert(std::is_same<special_storage_info_ty,
                      storage_info<0, layout_map<0, 1, -1>, halo<1, 2, 3>, alignment<1>>>::value,
        "");

    /*########## DATA STORE CHECKS ########## */
    typedef typename storage_traits_t::template data_store_t<double, storage_info_ty> data_store_t;
    static_assert(std::is_same<typename data_store_t::storage_info_t, storage_info_ty>::value, "");
    static_assert(std::is_same<typename data_store_t::data_t, double>::value, "");

    // storage check
    static_assert(std::is_same<typename data_store_t::storage_t, host_storage<double>>::value, "");
};

#ifdef __CUDACC__
// static type tests for Cuda backend
template <>
struct static_type_tests<backend::cuda> {
    using storage_traits_t = storage_traits<backend::cuda>;

    /*########## STORAGE INFO CHECKS ########## */
    // storage info check
    typedef storage_traits_t::storage_info_t<0, 3, halo<1, 2, 3>> storage_info_ty;
    static_assert(is_storage_info<storage_info_ty>::value, "");
    static_assert(
        std::is_same<storage_info_ty, storage_info<0, layout_map<2, 1, 0>, halo<1, 2, 3>, alignment<32>>>::value, "");

    // special layout
    typedef storage_traits_t::special_storage_info_t<0, selector<1, 1, 0>, halo<1, 2, 3>> special_storage_info_ty;
    static_assert(std::is_same<special_storage_info_ty,
                      storage_info<0, layout_map<1, 0, -1>, halo<1, 2, 3>, alignment<32>>>::value,
        "");

    /*########## DATA STORE CHECKS ########## */
    typedef storage_traits_t::data_store_t<double, storage_info_ty> data_store_t;
    static_assert(std::is_same<typename data_store_t::storage_info_t, storage_info_ty>::value, "");
    static_assert(std::is_same<typename data_store_t::data_t, double>::value, "");

    // storage check
    static_assert(std::is_same<typename data_store_t::storage_t, cuda_storage<double>>::value, "");
};
#endif

#ifndef GT_ICOSAHEDRAL_GRIDS
// static type tests for Mic backend
template <>
struct static_type_tests<backend::mc> {
    using storage_traits_t = storage_traits<backend::mc>;

    /*########## STORAGE INFO CHECKS ########## */
    // storage info check
    typedef storage_traits_t::storage_info_t<0, 3, halo<1, 2, 3>> storage_info_ty;
    static_assert(is_storage_info<storage_info_ty>::value, "");
    static_assert(
        std::is_same<storage_info_ty, storage_info<0, layout_map<2, 0, 1>, halo<1, 2, 3>, alignment<8>>>::value, "");

    // special layout
    typedef storage_traits_t::special_storage_info_t<0, selector<1, 1, 0>, halo<1, 2, 3>> special_storage_info_ty;
    static_assert(std::is_same<special_storage_info_ty,
                      storage_info<0, layout_map<1, 0, -1>, halo<1, 2, 3>, alignment<8>>>::value,
        "");

    /*########## DATA STORE CHECKS ########## */
    typedef storage_traits_t::data_store_t<double, storage_info_ty> data_store_t;
    static_assert(std::is_same<typename data_store_t::storage_info_t, storage_info_ty>::value, "");
    static_assert(std::is_same<typename data_store_t::data_t, double>::value, "");

    // storage check
    static_assert(std::is_same<typename data_store_t::storage_t, mc_storage<double>>::value, "");
};
#endif

#ifdef __CUDACC__
template <typename View>
__global__ void kernel(View v) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                v(i, j, k) *= 2;
}
#endif

TEST(StorageFacility, ViewTests) {
    typedef storage_traits<backend_t>::storage_info_t<0, 3> storage_info_ty;
    typedef storage_traits<backend_t>::data_store_t<double, storage_info_ty> data_store_t;

    // create a data_store_t
    storage_info_ty si(3, 3, 3);
    data_store_t ds(si);
    auto hv = make_host_view(ds);

    // fill with values
    uint_t x = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                hv(i, j, k) = x++;

    // sync
    ds.sync();

// do some computation
#ifdef __CUDACC__
    kernel<<<1, 1>>>(make_device_view(ds));
#else
    ds.reactivate_host_write_views();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                hv(i, j, k) *= 2;
#endif

    // sync
    ds.sync();

    // create a read only data view
    auto hrv = make_host_view<access_mode::read_only>(ds);

    // validate
    uint_t z = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                EXPECT_EQ(hrv(i, j, k), 2 * z++);
}

template <class Backend>
struct static_layout_test_cases;

template <class Backend>
struct static_layout_test_cases {
    using layout1_t = typename storage_traits<Backend>::template storage_info_t<0, 1>::layout_t;
    using layout2_t = typename storage_traits<Backend>::template storage_info_t<0, 2>::layout_t;
    using layout3_t = typename storage_traits<Backend>::template storage_info_t<0, 3>::layout_t;
    using layout4_t = typename storage_traits<Backend>::template storage_info_t<0, 4>::layout_t;
    using layout5_t = typename storage_traits<Backend>::template storage_info_t<0, 5>::layout_t;

    using layout_s5_t =
        typename storage_traits<Backend>::template special_storage_info_t<0, selector<1, 1, 1, 1, 1>>::layout_t;
    using layout_s51_t =
        typename storage_traits<Backend>::template special_storage_info_t<0, selector<0, 1, 1, 1, 1>>::layout_t;
    using layout_s52_t =
        typename storage_traits<Backend>::template special_storage_info_t<0, selector<1, 0, 1, 1, 1>>::layout_t;
    using layout_s53_t =
        typename storage_traits<Backend>::template special_storage_info_t<0, selector<1, 1, 0, 1, 1>>::layout_t;
    using layout_s54_t =
        typename storage_traits<Backend>::template special_storage_info_t<0, selector<1, 1, 1, 0, 1>>::layout_t;
    using layout_s55_t =
        typename storage_traits<Backend>::template special_storage_info_t<0, selector<1, 1, 1, 1, 0>>::layout_t;

    using layout_s56_t =
        typename storage_traits<Backend>::template special_storage_info_t<0, selector<0, 0, 1, 1, 1>>::layout_t;
    using layout_s57_t =
        typename storage_traits<Backend>::template special_storage_info_t<0, selector<1, 0, 0, 1, 1>>::layout_t;
    using layout_s58_t =
        typename storage_traits<Backend>::template special_storage_info_t<0, selector<1, 1, 0, 0, 1>>::layout_t;
    using layout_s59_t =
        typename storage_traits<Backend>::template special_storage_info_t<0, selector<1, 1, 1, 0, 0>>::layout_t;

    using layout_s510_t =
        typename storage_traits<Backend>::template special_storage_info_t<0, selector<0, 1, 0, 1, 1>>::layout_t;
    using layout_s511_t =
        typename storage_traits<Backend>::template special_storage_info_t<0, selector<1, 0, 1, 0, 1>>::layout_t;
    using layout_s512_t =
        typename storage_traits<Backend>::template special_storage_info_t<0, selector<1, 1, 0, 1, 0>>::layout_t;

    using layout_s513_t =
        typename storage_traits<Backend>::template special_storage_info_t<0, selector<0, 1, 1, 0, 1>>::layout_t;
    using layout_s514_t =
        typename storage_traits<Backend>::template special_storage_info_t<0, selector<1, 0, 1, 1, 0>>::layout_t;

    using layout_s515_t =
        typename storage_traits<Backend>::template special_storage_info_t<0, selector<0, 1, 1, 1, 0>>::layout_t;

    using layout_s516_t =
        typename storage_traits<Backend>::template special_storage_info_t<0, selector<0, 0, 0, 1, 1>>::layout_t;
    using layout_s517_t =
        typename storage_traits<Backend>::template special_storage_info_t<0, selector<1, 0, 0, 0, 1>>::layout_t;
    using layout_s518_t =
        typename storage_traits<Backend>::template special_storage_info_t<0, selector<1, 1, 0, 0, 0>>::layout_t;
    using layout_s519_t =
        typename storage_traits<Backend>::template special_storage_info_t<0, selector<0, 1, 1, 0, 0>>::layout_t;
    using layout_s520_t =
        typename storage_traits<Backend>::template special_storage_info_t<0, selector<0, 0, 1, 1, 0>>::layout_t;

    using layout_s521_t =
        typename storage_traits<Backend>::template special_storage_info_t<0, selector<1, 0, 0, 0, 0>>::layout_t;
    using layout_s522_t =
        typename storage_traits<Backend>::template special_storage_info_t<0, selector<0, 1, 0, 0, 0>>::layout_t;
    using layout_s523_t =
        typename storage_traits<Backend>::template special_storage_info_t<0, selector<0, 0, 1, 0, 0>>::layout_t;
    using layout_s524_t =
        typename storage_traits<Backend>::template special_storage_info_t<0, selector<0, 0, 0, 1, 0>>::layout_t;
    using layout_s525_t =
        typename storage_traits<Backend>::template special_storage_info_t<0, selector<0, 0, 0, 0, 1>>::layout_t;
};

template <class Backend>
struct static_layout_tests_decreasing : static_layout_test_cases<Backend> {
    using cases = static_layout_test_cases<Backend>;
    static_assert(std::is_same<typename cases::layout1_t, layout_map<0>>::value, "");
    static_assert(std::is_same<typename cases::layout2_t, layout_map<1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout3_t, layout_map<2, 1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout4_t, layout_map<3, 2, 1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout5_t, layout_map<4, 3, 2, 1, 0>>::value, "");

    static_assert(std::is_same<typename cases::layout_s5_t, layout_map<4, 3, 2, 1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s51_t, layout_map<-1, 3, 2, 1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s52_t, layout_map<3, -1, 2, 1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s53_t, layout_map<3, 2, -1, 1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s54_t, layout_map<3, 2, 1, -1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s55_t, layout_map<3, 2, 1, 0, -1>>::value, "");

    static_assert(std::is_same<typename cases::layout_s56_t, layout_map<-1, -1, 2, 1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s57_t, layout_map<2, -1, -1, 1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s58_t, layout_map<2, 1, -1, -1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s59_t, layout_map<2, 1, 0, -1, -1>>::value, "");

    static_assert(std::is_same<typename cases::layout_s510_t, layout_map<-1, 2, -1, 1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s511_t, layout_map<2, -1, 1, -1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s512_t, layout_map<2, 1, -1, 0, -1>>::value, "");

    static_assert(std::is_same<typename cases::layout_s513_t, layout_map<-1, 2, 1, -1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s514_t, layout_map<2, -1, 1, 0, -1>>::value, "");

    static_assert(std::is_same<typename cases::layout_s515_t, layout_map<-1, 2, 1, 0, -1>>::value, "");

    static_assert(std::is_same<typename cases::layout_s516_t, layout_map<-1, -1, -1, 1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s517_t, layout_map<1, -1, -1, -1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s518_t, layout_map<1, 0, -1, -1, -1>>::value, "");
    static_assert(std::is_same<typename cases::layout_s519_t, layout_map<-1, 1, 0, -1, -1>>::value, "");
    static_assert(std::is_same<typename cases::layout_s520_t, layout_map<-1, -1, 1, 0, -1>>::value, "");

    static_assert(std::is_same<typename cases::layout_s521_t, layout_map<0, -1, -1, -1, -1>>::value, "");
    static_assert(std::is_same<typename cases::layout_s522_t, layout_map<-1, 0, -1, -1, -1>>::value, "");
    static_assert(std::is_same<typename cases::layout_s523_t, layout_map<-1, -1, 0, -1, -1>>::value, "");
    static_assert(std::is_same<typename cases::layout_s524_t, layout_map<-1, -1, -1, 0, -1>>::value, "");
    static_assert(std::is_same<typename cases::layout_s525_t, layout_map<-1, -1, -1, -1, 0>>::value, "");
};

template <class Backend>
struct static_layout_tests_decreasing_swappedxy : static_layout_test_cases<Backend> {
    using cases = static_layout_test_cases<Backend>;
    static_assert(std::is_same<typename cases::layout1_t, layout_map<0>>::value, "");
    static_assert(std::is_same<typename cases::layout2_t, layout_map<1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout3_t, layout_map<2, 0, 1>>::value, "");
    static_assert(std::is_same<typename cases::layout4_t, layout_map<3, 1, 2, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout5_t, layout_map<4, 2, 3, 1, 0>>::value, "");

    static_assert(std::is_same<typename cases::layout_s5_t, layout_map<4, 2, 3, 1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s51_t, layout_map<-1, 2, 3, 1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s52_t, layout_map<3, -1, 2, 1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s53_t, layout_map<3, 2, -1, 1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s54_t, layout_map<3, 1, 2, -1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s55_t, layout_map<3, 1, 2, 0, -1>>::value, "");

    static_assert(std::is_same<typename cases::layout_s56_t, layout_map<-1, -1, 2, 1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s57_t, layout_map<2, -1, -1, 1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s58_t, layout_map<2, 1, -1, -1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s59_t, layout_map<2, 0, 1, -1, -1>>::value, "");

    static_assert(std::is_same<typename cases::layout_s510_t, layout_map<-1, 2, -1, 1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s511_t, layout_map<2, -1, 1, -1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s512_t, layout_map<2, 1, -1, 0, -1>>::value, "");

    static_assert(std::is_same<typename cases::layout_s513_t, layout_map<-1, 1, 2, -1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s514_t, layout_map<2, -1, 1, 0, -1>>::value, "");

    static_assert(std::is_same<typename cases::layout_s515_t, layout_map<-1, 1, 2, 0, -1>>::value, "");

    static_assert(std::is_same<typename cases::layout_s516_t, layout_map<-1, -1, -1, 1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s517_t, layout_map<1, -1, -1, -1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s518_t, layout_map<1, 0, -1, -1, -1>>::value, "");
    static_assert(std::is_same<typename cases::layout_s519_t, layout_map<-1, 0, 1, -1, -1>>::value, "");
    static_assert(std::is_same<typename cases::layout_s520_t, layout_map<-1, -1, 1, 0, -1>>::value, "");

    static_assert(std::is_same<typename cases::layout_s521_t, layout_map<0, -1, -1, -1, -1>>::value, "");
    static_assert(std::is_same<typename cases::layout_s522_t, layout_map<-1, 0, -1, -1, -1>>::value, "");
    static_assert(std::is_same<typename cases::layout_s523_t, layout_map<-1, -1, 0, -1, -1>>::value, "");
    static_assert(std::is_same<typename cases::layout_s524_t, layout_map<-1, -1, -1, 0, -1>>::value, "");
    static_assert(std::is_same<typename cases::layout_s525_t, layout_map<-1, -1, -1, -1, 0>>::value, "");
};

template <class Backend>
struct static_layout_tests_increasing : static_layout_test_cases<Backend> {
    using cases = static_layout_test_cases<Backend>;
    static_assert(std::is_same<typename cases::layout1_t, layout_map<0>>::value, "");
    static_assert(std::is_same<typename cases::layout2_t, layout_map<0, 1>>::value, "");
    static_assert(std::is_same<typename cases::layout3_t, layout_map<0, 1, 2>>::value, "");
    static_assert(std::is_same<typename cases::layout4_t, layout_map<1, 2, 3, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout5_t, layout_map<2, 3, 4, 0, 1>>::value, "");

    static_assert(std::is_same<typename cases::layout_s5_t, layout_map<2, 3, 4, 0, 1>>::value, "");
    static_assert(std::is_same<typename cases::layout_s51_t, layout_map<-1, 2, 3, 0, 1>>::value, "");
    static_assert(std::is_same<typename cases::layout_s52_t, layout_map<2, -1, 3, 0, 1>>::value, "");
    static_assert(std::is_same<typename cases::layout_s53_t, layout_map<2, 3, -1, 0, 1>>::value, "");
    static_assert(std::is_same<typename cases::layout_s54_t, layout_map<1, 2, 3, -1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s55_t, layout_map<1, 2, 3, 0, -1>>::value, "");

    static_assert(std::is_same<typename cases::layout_s56_t, layout_map<-1, -1, 2, 0, 1>>::value, "");
    static_assert(std::is_same<typename cases::layout_s57_t, layout_map<2, -1, -1, 0, 1>>::value, "");
    static_assert(std::is_same<typename cases::layout_s58_t, layout_map<1, 2, -1, -1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s59_t, layout_map<0, 1, 2, -1, -1>>::value, "");

    static_assert(std::is_same<typename cases::layout_s510_t, layout_map<-1, 2, -1, 0, 1>>::value, "");
    static_assert(std::is_same<typename cases::layout_s511_t, layout_map<1, -1, 2, -1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s512_t, layout_map<1, 2, -1, 0, -1>>::value, "");

    static_assert(std::is_same<typename cases::layout_s513_t, layout_map<-1, 1, 2, -1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s514_t, layout_map<1, -1, 2, 0, -1>>::value, "");

    static_assert(std::is_same<typename cases::layout_s515_t, layout_map<-1, 1, 2, 0, -1>>::value, "");

    static_assert(std::is_same<typename cases::layout_s516_t, layout_map<-1, -1, -1, 0, 1>>::value, "");
    static_assert(std::is_same<typename cases::layout_s517_t, layout_map<1, -1, -1, -1, 0>>::value, "");
    static_assert(std::is_same<typename cases::layout_s518_t, layout_map<0, 1, -1, -1, -1>>::value, "");
    static_assert(std::is_same<typename cases::layout_s519_t, layout_map<-1, 0, 1, -1, -1>>::value, "");
    static_assert(std::is_same<typename cases::layout_s520_t, layout_map<-1, -1, 1, 0, -1>>::value, "");

    static_assert(std::is_same<typename cases::layout_s521_t, layout_map<0, -1, -1, -1, -1>>::value, "");
    static_assert(std::is_same<typename cases::layout_s522_t, layout_map<-1, 0, -1, -1, -1>>::value, "");
    static_assert(std::is_same<typename cases::layout_s523_t, layout_map<-1, -1, 0, -1, -1>>::value, "");
    static_assert(std::is_same<typename cases::layout_s524_t, layout_map<-1, -1, -1, 0, -1>>::value, "");
    static_assert(std::is_same<typename cases::layout_s525_t, layout_map<-1, -1, -1, -1, 0>>::value, "");
};

template <typename Backend>
struct static_layout_tests {
    static_assert(sizeof(Backend) > 0, "");
};

#ifdef __CUDACC__
template <>
struct static_layout_tests<backend::cuda> : static_layout_tests_decreasing<backend::cuda> {};
#endif

#ifndef GT_ICOSAHEDRAL_GRIDS
template <>
struct static_layout_tests<backend::mc> : static_layout_tests_decreasing_swappedxy<backend::mc> {};
#endif

template <>
struct static_layout_tests<backend::x86> : static_layout_tests_increasing<backend::x86> {};

template <>
struct static_layout_tests<backend::naive> : static_layout_tests_increasing<backend::naive> {};

static_assert(
    std::is_same<typename storage_traits<backend_t>::custom_layout_storage_info_t<0, layout_map<2, 1, 0>>::layout_t,
        layout_map<2, 1, 0>>::value,
    "");
static_assert(std::is_same<storage_traits<backend_t>::custom_layout_storage_info_t<0, layout_map<1, 0>>::layout_t,
                  layout_map<1, 0>>::value,
    "");
static_assert(std::is_same<typename storage_traits<backend_t>::custom_layout_storage_info_t<0, layout_map<0>>::layout_t,
                  layout_map<0>>::value,
    "");
static_assert(
    std::is_same<typename storage_traits<backend_t>::custom_layout_storage_info_t<0, layout_map<2, -1, 1, 0>>::layout_t,
        layout_map<2, -1, 1, 0>>::value,
    "");

TEST(dummy, dummy) {}

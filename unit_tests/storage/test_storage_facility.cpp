/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/

#include "gtest/gtest.h"

#include <boost/mpl/int.hpp>
#include <boost/type_traits.hpp>

#include <storage/storage-facility.hpp>
#include <common/gt_assert.hpp>

using namespace gridtools;

#ifdef __CUDACC__
#define BACKEND enumtype::Cuda
#else
#define BACKEND enumtype::Host
#endif

TEST(StorageFacility, TypesTest) {
/*########## STORAGE INFO CHECKS ########## */
#ifdef __CUDACC__
    // storage info check
    typedef storage_traits< BACKEND >::storage_info_t< 0, 3, halo< 1, 2, 3 > > storage_info_ty;
    GRIDTOOLS_STATIC_ASSERT(
        (is_storage_info< storage_info_ty >::type::value), "is_storage_info metafunction is not working anymore");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< storage_info_ty,
            cuda_storage_info< 0, layout_map< 2, 1, 0 >, halo< 1, 2, 3 >, alignment< 32 > > >::type::value),
        "storage info test failed");

    // special layout
    typedef storage_traits< BACKEND >::special_storage_info_t< 0, selector< 1, 1, 0 >, halo< 1, 2, 3 > >
        special_storage_info_ty;
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< special_storage_info_ty,
            cuda_storage_info< 0, layout_map< 1, 0, -1 >, halo< 1, 2, 3 >, alignment< 32 > > >::type::value),
        "storage info test failed");
#else
    // storage info check
    typedef storage_traits< BACKEND >::storage_info_t< 0, 3, halo< 1, 2, 3 > > storage_info_ty;
    GRIDTOOLS_STATIC_ASSERT(
        (is_storage_info< storage_info_ty >::type::value), "is_storage_info metafunction is not working anymore");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< storage_info_ty,
            host_storage_info< 0, layout_map< 0, 1, 2 >, halo< 1, 2, 3 >, alignment< 1 > > >::type::value),
        "storage info test failed");

    // special layout
    typedef storage_traits< BACKEND >::special_storage_info_t< 0, selector< 1, 1, 0 >, halo< 1, 2, 3 > >
        special_storage_info_ty;
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< special_storage_info_ty,
            host_storage_info< 0, layout_map< 0, 1, -1 >, halo< 1, 2, 3 >, alignment< 1 > > >::type::value),
        "storage info test failed");
#endif

    /*########## DATA STORE CHECKS ########## */
    typedef storage_traits< BACKEND >::data_store_t< double, storage_info_ty > data_store_t;
    // data store check (data_t is common, storage_info_t was typedefed before)
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< typename data_store_t::storage_info_t, storage_info_ty >::type::value),
        "data store info type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< typename data_store_t::data_t, double >::type::value), "data store value type is wrong");

#ifdef __CUDACC__
    // storage check
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< typename data_store_t::storage_t, cuda_storage< double > >::type::value),
        "data store storage type is wrong");
#else
    // storage check
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< typename data_store_t::storage_t, host_storage< double > >::type::value),
        "data store storage type is wrong");
#endif

    /*########## DATA STORE FIELD CHECKS ########## */
    typedef storage_traits< BACKEND >::data_store_field_t< double, storage_info_ty, 1, 2, 3 > data_store_field_t;
    // data store check (data_t is common, storage_info_t was typedefed before)
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< typename data_store_field_t::storage_info_t, storage_info_ty >::type::value),
        "data store field info type is wrong");
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< typename data_store_field_t::data_store_t, data_store_t >::type::value),
        "internal data store type of data store field type is wrong");
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< typename data_store_field_t::data_t, double >::type::value),
        "data store field value type is wrong");
}

#ifdef __CUDACC__
template < typename View >
__global__ void kernel(View v) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                v(i, j, k) *= 2;
}
#endif

TEST(StorageFacility, ViewTests) {
    typedef storage_traits< BACKEND >::storage_info_t< 0, 3 > storage_info_ty;
    typedef storage_traits< BACKEND >::data_store_t< double, storage_info_ty > data_store_t;
    typedef storage_traits< BACKEND >::data_store_field_t< double, storage_info_ty, 1, 2, 3 > data_store_field_t;

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
    kernel<<< 1, 1 >>>(make_device_view(ds));
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
    auto hrv = make_host_view< access_mode::ReadOnly >(ds);

    // validate
    uint_t z = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                EXPECT_EQ(hrv(i, j, k), 2 * z++);
}

TEST(StorageFacility, LayoutTests) {
    typedef typename storage_traits< BACKEND >::storage_info_t< 0, 1 >::layout_t layout1_t;
    typedef typename storage_traits< BACKEND >::storage_info_t< 0, 2 >::layout_t layout2_t;
    typedef typename storage_traits< BACKEND >::storage_info_t< 0, 3 >::layout_t layout3_t;
    typedef typename storage_traits< BACKEND >::storage_info_t< 0, 4 >::layout_t layout4_t;
    typedef typename storage_traits< BACKEND >::storage_info_t< 0, 5 >::layout_t layout5_t;

    typedef typename storage_traits< BACKEND >::special_storage_info_t< 0, selector< 1, 1, 1, 1, 1 > >::layout_t
        layout_s5_t;
    typedef typename storage_traits< BACKEND >::special_storage_info_t< 0, selector< 0, 1, 1, 1, 1 > >::layout_t
        layout_s51_t;
    typedef typename storage_traits< BACKEND >::special_storage_info_t< 0, selector< 1, 0, 1, 1, 1 > >::layout_t
        layout_s52_t;
    typedef typename storage_traits< BACKEND >::special_storage_info_t< 0, selector< 1, 1, 0, 1, 1 > >::layout_t
        layout_s53_t;
    typedef typename storage_traits< BACKEND >::special_storage_info_t< 0, selector< 1, 1, 1, 0, 1 > >::layout_t
        layout_s54_t;
    typedef typename storage_traits< BACKEND >::special_storage_info_t< 0, selector< 1, 1, 1, 1, 0 > >::layout_t
        layout_s55_t;

    typedef typename storage_traits< BACKEND >::special_storage_info_t< 0, selector< 0, 0, 1, 1, 1 > >::layout_t
        layout_s56_t;
    typedef typename storage_traits< BACKEND >::special_storage_info_t< 0, selector< 1, 0, 0, 1, 1 > >::layout_t
        layout_s57_t;
    typedef typename storage_traits< BACKEND >::special_storage_info_t< 0, selector< 1, 1, 0, 0, 1 > >::layout_t
        layout_s58_t;
    typedef typename storage_traits< BACKEND >::special_storage_info_t< 0, selector< 1, 1, 1, 0, 0 > >::layout_t
        layout_s59_t;

    typedef typename storage_traits< BACKEND >::special_storage_info_t< 0, selector< 0, 1, 0, 1, 1 > >::layout_t
        layout_s510_t;
    typedef typename storage_traits< BACKEND >::special_storage_info_t< 0, selector< 1, 0, 1, 0, 1 > >::layout_t
        layout_s511_t;
    typedef typename storage_traits< BACKEND >::special_storage_info_t< 0, selector< 1, 1, 0, 1, 0 > >::layout_t
        layout_s512_t;

    typedef typename storage_traits< BACKEND >::special_storage_info_t< 0, selector< 0, 1, 1, 0, 1 > >::layout_t
        layout_s513_t;
    typedef typename storage_traits< BACKEND >::special_storage_info_t< 0, selector< 1, 0, 1, 1, 0 > >::layout_t
        layout_s514_t;

    typedef typename storage_traits< BACKEND >::special_storage_info_t< 0, selector< 0, 1, 1, 1, 0 > >::layout_t
        layout_s515_t;

    typedef typename storage_traits< BACKEND >::special_storage_info_t< 0, selector< 0, 0, 0, 1, 1 > >::layout_t
        layout_s516_t;
    typedef typename storage_traits< BACKEND >::special_storage_info_t< 0, selector< 1, 0, 0, 0, 1 > >::layout_t
        layout_s517_t;
    typedef typename storage_traits< BACKEND >::special_storage_info_t< 0, selector< 1, 1, 0, 0, 0 > >::layout_t
        layout_s518_t;
    typedef typename storage_traits< BACKEND >::special_storage_info_t< 0, selector< 0, 1, 1, 0, 0 > >::layout_t
        layout_s519_t;
    typedef typename storage_traits< BACKEND >::special_storage_info_t< 0, selector< 0, 0, 1, 1, 0 > >::layout_t
        layout_s520_t;

    typedef typename storage_traits< BACKEND >::special_storage_info_t< 0, selector< 1, 0, 0, 0, 0 > >::layout_t
        layout_s521_t;
    typedef typename storage_traits< BACKEND >::special_storage_info_t< 0, selector< 0, 1, 0, 0, 0 > >::layout_t
        layout_s522_t;
    typedef typename storage_traits< BACKEND >::special_storage_info_t< 0, selector< 0, 0, 1, 0, 0 > >::layout_t
        layout_s523_t;
    typedef typename storage_traits< BACKEND >::special_storage_info_t< 0, selector< 0, 0, 0, 1, 0 > >::layout_t
        layout_s524_t;
    typedef typename storage_traits< BACKEND >::special_storage_info_t< 0, selector< 0, 0, 0, 0, 1 > >::layout_t
        layout_s525_t;
#ifdef __CUDACC__
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< layout1_t, layout_map< 0 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< layout2_t, layout_map< 1, 0 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< layout3_t, layout_map< 2, 1, 0 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< layout4_t, layout_map< 3, 2, 1, 0 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< layout5_t, layout_map< 4, 3, 2, 1, 0 > >::value), "layout type is wrong");

    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s5_t, layout_map< 4, 3, 2, 1, 0 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s51_t, layout_map< -1, 3, 2, 1, 0 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s52_t, layout_map< 3, -1, 2, 1, 0 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s53_t, layout_map< 3, 2, -1, 1, 0 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s54_t, layout_map< 3, 2, 1, -1, 0 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s55_t, layout_map< 3, 2, 1, 0, -1 > >::value), "layout type is wrong");

    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s56_t, layout_map< -1, -1, 2, 1, 0 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s57_t, layout_map< 2, -1, -1, 1, 0 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s58_t, layout_map< 2, 1, -1, -1, 0 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s59_t, layout_map< 2, 1, 0, -1, -1 > >::value), "layout type is wrong");

    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s510_t, layout_map< -1, 2, -1, 1, 0 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s511_t, layout_map< 2, -1, 1, -1, 0 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s512_t, layout_map< 2, 1, -1, 0, -1 > >::value), "layout type is wrong");

    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s513_t, layout_map< -1, 2, 1, -1, 0 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s514_t, layout_map< 2, -1, 1, 0, -1 > >::value), "layout type is wrong");

    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s515_t, layout_map< -1, 2, 1, 0, -1 > >::value), "layout type is wrong");

    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s516_t, layout_map< -1, -1, -1, 1, 0 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s517_t, layout_map< 1, -1, -1, -1, 0 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s518_t, layout_map< 1, 0, -1, -1, -1 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s519_t, layout_map< -1, 1, 0, -1, -1 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s520_t, layout_map< -1, -1, 1, 0, -1 > >::value), "layout type is wrong");

    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s521_t, layout_map< 0, -1, -1, -1, -1 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s522_t, layout_map< -1, 0, -1, -1, -1 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s523_t, layout_map< -1, -1, 0, -1, -1 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s524_t, layout_map< -1, -1, -1, 0, -1 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s525_t, layout_map< -1, -1, -1, -1, 0 > >::value), "layout type is wrong");
#else
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< layout1_t, layout_map< 0 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< layout2_t, layout_map< 0, 1 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< layout3_t, layout_map< 0, 1, 2 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< layout4_t, layout_map< 1, 2, 3, 0 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< layout5_t, layout_map< 2, 3, 4, 0, 1 > >::value), "layout type is wrong");

    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s5_t, layout_map< 2, 3, 4, 0, 1 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s51_t, layout_map< -1, 2, 3, 0, 1 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s52_t, layout_map< 2, -1, 3, 0, 1 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s53_t, layout_map< 2, 3, -1, 0, 1 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s54_t, layout_map< 1, 2, 3, -1, 0 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s55_t, layout_map< 1, 2, 3, 0, -1 > >::value), "layout type is wrong");

    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s56_t, layout_map< -1, -1, 2, 0, 1 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s57_t, layout_map< 2, -1, -1, 0, 1 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s58_t, layout_map< 1, 2, -1, -1, 0 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s59_t, layout_map< 0, 1, 2, -1, -1 > >::value), "layout type is wrong");

    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s510_t, layout_map< -1, 2, -1, 0, 1 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s511_t, layout_map< 1, -1, 2, -1, 0 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s512_t, layout_map< 1, 2, -1, 0, -1 > >::value), "layout type is wrong");

    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s513_t, layout_map< -1, 1, 2, -1, 0 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s514_t, layout_map< 1, -1, 2, 0, -1 > >::value), "layout type is wrong");

    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s515_t, layout_map< -1, 1, 2, 0, -1 > >::value), "layout type is wrong");

    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s516_t, layout_map< -1, -1, -1, 0, 1 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s517_t, layout_map< 1, -1, -1, -1, 0 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s518_t, layout_map< 0, 1, -1, -1, -1 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s519_t, layout_map< -1, 0, 1, -1, -1 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s520_t, layout_map< -1, -1, 1, 0, -1 > >::value), "layout type is wrong");

    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s521_t, layout_map< 0, -1, -1, -1, -1 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s522_t, layout_map< -1, 0, -1, -1, -1 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s523_t, layout_map< -1, -1, 0, -1, -1 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s524_t, layout_map< -1, -1, -1, 0, -1 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< layout_s525_t, layout_map< -1, -1, -1, -1, 0 > >::value), "layout type is wrong");
#endif
}

TEST(StorageFacility, CustomLayoutTests) {
    typedef typename storage_traits< BACKEND >::custom_layout_storage_info_t< 0, layout_map< 2, 1, 0 > >::layout_t
        layout3_t;
    typedef
        typename storage_traits< BACKEND >::custom_layout_storage_info_t< 0, layout_map< 1, 0 > >::layout_t layout2_t;
    typedef typename storage_traits< BACKEND >::custom_layout_storage_info_t< 0, layout_map< 0 > >::layout_t layout1_t;
    typedef typename storage_traits< BACKEND >::custom_layout_storage_info_t< 0, layout_map< 2, -1, 1, 0 > >::layout_t
        layout4_t;
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< layout3_t, layout_map< 2, 1, 0 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< layout2_t, layout_map< 1, 0 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< layout1_t, layout_map< 0 > >::value), "layout type is wrong");
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< layout4_t, layout_map< 2, -1, 1, 0 > >::value), "layout type is wrong");
}

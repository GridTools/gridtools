/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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

#include <boost/type_traits.hpp>

#include "storage-facility.hpp"

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
    typedef storage_traits< BACKEND >::storage_info_t< 0, 3 > storage_info_ty;
    static_assert(
        (is_storage_info< storage_info_ty >::type::value), "is_storage_info metafunction is not working anymore");
    static_assert((boost::is_same< storage_info_ty, cuda_storage_info< 0, layout_map< 2, 1, 0 > > >::type::value),
        "storage info test failed");

    // special layout
    typedef storage_traits< BACKEND >::special_storage_info_t< 0, selector< 1, 1, 0 > > special_storage_info_ty;
    static_assert(
        (boost::is_same< special_storage_info_ty, cuda_storage_info< 0, layout_map< 1, 0, -1 > > >::type::value),
        "storage info test failed");
#else
    // storage info check
    typedef storage_traits< BACKEND >::storage_info_t< 0, 3 > storage_info_ty;
    static_assert(
        (is_storage_info< storage_info_ty >::type::value), "is_storage_info metafunction is not working anymore");
    static_assert((boost::is_same< storage_info_ty, host_storage_info< 0, layout_map< 0, 1, 2 > > >::type::value),
        "storage info test failed");

    // special layout
    typedef storage_traits< BACKEND >::special_storage_info_t< 0, selector< 1, 1, 0 > > special_storage_info_ty;
    static_assert(
        (boost::is_same< special_storage_info_ty, host_storage_info< 0, layout_map< 0, 1, -1 > > >::type::value),
        "storage info test failed");
#endif

    /*########## DATA STORE CHECKS ########## */
    typedef storage_traits< BACKEND >::data_store_t< double, storage_info_ty > data_store_t;
    // data store check (data_t is common, storage_info_t was typedefed before)
    static_assert(boost::is_same< typename data_store_t::storage_info_t, storage_info_ty >::type::value,
        "data store info type is wrong");
    static_assert(
        boost::is_same< typename data_store_t::data_t, double >::type::value, "data store value type is wrong");

#ifdef __CUDACC__
    // storage check
    static_assert(boost::is_same< typename data_store_t::storage_t, cuda_storage< double > >::type::value,
        "data store storage type is wrong");
#else
    // storage check
    static_assert(boost::is_same< typename data_store_t::storage_t, host_storage< double > >::type::value,
        "data store storage type is wrong");
#endif

    /*########## DATA STORE FIELD CHECKS ########## */
    typedef storage_traits< BACKEND >::data_store_field_t< double, storage_info_ty, 1, 2, 3 > data_store_field_t;
    // data store check (data_t is common, storage_info_t was typedefed before)
    static_assert(boost::is_same< typename data_store_field_t::storage_info_t, storage_info_ty >::type::value,
        "data store field info type is wrong");
    static_assert(boost::is_same< typename data_store_field_t::data_store_t, data_store_t >::type::value,
        "internal data store type of data store field type is wrong");
    static_assert(boost::is_same< typename data_store_field_t::data_t, double >::type::value,
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
    ds.allocate();
    auto hv = make_host_view(ds);

    // fill with values
    unsigned x = 0;
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
    auto hrv = make_host_view< true >(ds);

    // validate
    unsigned z = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                EXPECT_EQ(hrv(i, j, k), 2 * z++);
}

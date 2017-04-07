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

// i know that the following directive is super ugly,
// but i need to check the private member fields of
// the storage.
#define protected public
#include <common/defs.hpp>
#include <storage/storage-facility.hpp>

#ifdef _USE_GPU_
#include <storage/hybrid_pointer.hpp>
#else
#include <storage/wrap_pointer.hpp>
#endif

using namespace gridtools;

TEST(cuda_storage_on_host, test_storage_types) {
    typedef layout_map< 0, 1, 2 > layout;
#ifdef __CUDACC__
#define BACKEND enumtype::Cuda
#else
#define BACKEND enumtype::Host
#endif

#ifdef CXX11_ENABLED
    typedef storage_traits< BACKEND >::meta_storage_type< 0, layout > meta_data_t;
    typedef storage_traits< BACKEND >::storage_type< float, meta_data_t > storage_t;
#else
    typedef storage_traits< BACKEND >::meta_storage_type< 0, layout >::type meta_data_t;
    typedef storage_traits< BACKEND >::storage_type< float, meta_data_t >::type storage_t;
#endif
    meta_data_t meta_obj(10, 10, 10);
    storage_t st_obj(meta_obj, "in");
#ifdef __CUDACC__
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< meta_data_t,
            meta_storage< meta_storage_aligned< meta_storage_base< static_uint< 0 >, layout, false >,
                aligned< 32 >,
                halo< 0, 0, 0 > > > >::value),
        "type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< storage_t, storage< base_storage< hybrid_pointer< float >, meta_data_t, 1 > > >::value),
        "type is wrong");
#else
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< meta_data_t,
            meta_storage< meta_storage_aligned< meta_storage_base< static_uint< 0 >, layout, false >,
                aligned< 0 >,
                halo< 0, 0, 0 > > > >::value),
        "type is wrong");
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< storage_t, storage< base_storage< wrap_pointer< float >, meta_data_t, 1 > > >::value),
        "type is wrong");
#endif
}

TEST(cuda_storage_on_host, test_storage) {
    using namespace gridtools;
    using namespace enumtype;
    // some typedefs to create a storage.
    // either a host backend storage or a
    // cuda backend storage.
    typedef gridtools::layout_map< 0, 1, 2 > layout_t;
    typedef meta_storage<
        meta_storage_aligned< meta_storage_base< static_int< 0 >, layout_t, false >, aligned< 32 >, halo< 0, 0, 0 > > >
        meta_data_t;
#ifdef _USE_GPU_
    typedef base_storage< hybrid_pointer< double >, meta_data_t, 1 > base_st;
#else
    typedef base_storage< wrap_pointer< double >, meta_data_t, 1 > base_st;
#endif
    typedef storage< base_st > storage_t;
    // initializer the meta_data and the storage
    meta_data_t meta_data(10, 10, 10);
    storage_t foo_field(meta_data);
    // fill storage
    int z = 0;
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 10; ++j)
            for (int k = 0; k < 10; ++k)
                foo_field(i, j, k) = z;
    // get storage ptr and on_host information
    ASSERT_TRUE(foo_field.m_on_host && "The storage should not be located on the device.");
    base_st *ptr1 = foo_field.m_storage.get_pointer_to_use();
#ifdef _USE_GPU_
    // copy the field to the gpu and check
    // for correct behaviour.
    foo_field.h2d_update();
    base_st *ptr2 = foo_field.m_storage.get_pointer_to_use();
    ASSERT_FALSE(foo_field.m_on_host && "The storage should be located on the device.");
    ASSERT_TRUE(ptr1 != ptr2 && "Pointers to the storage must not be the same.");
    // copy the field back from the gpu
    foo_field.d2h_update();
#endif
    // check if the pointers are right and the
    // field is on the host again.
    base_st *ptr3 = foo_field.m_storage.get_pointer_to_use();
    ASSERT_TRUE(ptr1 == ptr3 && "Pointers to the storage must be the same.");
    ASSERT_TRUE(foo_field.m_on_host && "The storage should not be located on the device.");
}

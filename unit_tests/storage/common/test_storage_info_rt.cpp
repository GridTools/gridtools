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

#include "storage/common/storage_info_rt.hpp"
#include "storage/storage-facility.hpp"

using namespace gridtools;

#ifdef __CUDACC__
static constexpr enumtype::platform platform_ = enumtype::Cuda;
#else
static constexpr enumtype::platform platform_ = enumtype::Host;
#endif

TEST(StorageInfoRT, Make3D) {
    using storage_info_t = storage_traits< platform_ >::storage_info_t< 0, 3 >;
    storage_info_t si(4, 5, 6);

    auto storage_info_rt_ = make_storage_info_rt(si);

    auto dims = storage_info_rt_.dims();
    ASSERT_EQ(si.dim< 0 >(), dims[0]);
    ASSERT_EQ(si.dim< 1 >(), dims[1]);
    ASSERT_EQ(si.dim< 2 >(), dims[2]);

    auto unaligned_dims = storage_info_rt_.unaligned_dims();
    ASSERT_EQ(si.total_length< 0 >(), unaligned_dims[0]);
    ASSERT_EQ(si.total_length< 1 >(), unaligned_dims[1]);
    ASSERT_EQ(si.total_length< 2 >(), unaligned_dims[2]);

    auto strides = storage_info_rt_.strides();
    ASSERT_EQ(si.stride< 0 >(), strides[0]);
    ASSERT_EQ(si.stride< 1 >(), strides[1]);
    ASSERT_EQ(si.stride< 2 >(), strides[2]);

}

TEST(StorageInfoRT, Make3Dmasked) {
    using storage_info_t = storage_traits< platform_ >::special_storage_info_t< 0, selector< 1, 0, 1 > >;
    storage_info_t si(4, 5, 6);

    auto storage_info_rt_ = make_storage_info_rt(si);

    auto dims = storage_info_rt_.dims();
    ASSERT_EQ(si.dim< 0 >(), dims[0]);
    ASSERT_EQ(si.dim< 1 >(), dims[1]);
    ASSERT_EQ(si.dim< 2 >(), dims[2]);

    auto unaligned_dims = storage_info_rt_.unaligned_dims();
    ASSERT_EQ(si.total_length< 0 >(), unaligned_dims[0]);
    ASSERT_EQ(si.total_length< 1 >(), unaligned_dims[1]);
    ASSERT_EQ(si.total_length< 2 >(), unaligned_dims[2]);

    auto strides = storage_info_rt_.strides();
    ASSERT_EQ(si.stride< 0 >(), strides[0]);
    ASSERT_EQ(si.stride< 1 >(), strides[1]);
    ASSERT_EQ(si.stride< 2 >(), strides[2]);

}

/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "test_storage_facility.cpp"

TEST(StorageFacility, TestInCudaFile) {
    // just a small test that instantiates storage_info_t and data_store_t
    typedef gridtools::storage_traits<gridtools::backend::cuda>::storage_info_t<0, 1> storage_info_ty;
    storage_info_ty a(3);
    typedef gridtools::storage_traits<gridtools::backend::cuda>::data_store_t<double, storage_info_ty> data_store_t;
    data_store_t b(a);
}

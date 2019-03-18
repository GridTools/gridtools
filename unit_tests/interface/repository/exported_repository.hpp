/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/interface/repository/repository.hpp>
#include <gridtools/storage/storage_facility.hpp>
#include <gridtools/tools/backend_select.hpp>

using IJKStorageInfo = typename gridtools::storage_traits<gridtools::target::x86>::storage_info_t<0, 3>;
using IJKDataStore =
    typename gridtools::storage_traits<gridtools::target::x86>::data_store_t<float_type, IJKStorageInfo>;
using IJStorageInfo =
    typename gridtools::storage_traits<gridtools::target::x86>::special_storage_info_t<1, gridtools::selector<1, 1, 0>>;
using IJDataStore = typename gridtools::storage_traits<gridtools::target::x86>::data_store_t<float_type, IJStorageInfo>;
using JKStorageInfo =
    typename gridtools::storage_traits<gridtools::target::x86>::special_storage_info_t<2, gridtools::selector<0, 1, 1>>;
using JKDataStore = typename gridtools::storage_traits<gridtools::target::x86>::data_store_t<float_type, JKStorageInfo>;

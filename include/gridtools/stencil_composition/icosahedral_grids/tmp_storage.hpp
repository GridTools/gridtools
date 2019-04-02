/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "../../common/defs.hpp"

namespace gridtools {
    namespace tmp_storage {
        template <class StorageInfo, size_t NColors, class Backend>
        StorageInfo make_storage_info(Backend const &, size_t i, size_t j, size_t k) {
            return StorageInfo{i, NColors, j, k};
        }
    } // namespace tmp_storage
} // namespace gridtools

/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "../../common/defs.hpp"
#include "../backend_ids.hpp"

#include "./backend_mc/tmp_storage.hpp"

namespace gridtools {
    namespace tmp_storage {
        template <class StorageInfo, size_t /*NColors*/, class Platform, class Strategy>
        StorageInfo make_storage_info(
            backend_ids<Platform, grid_type::structured, Strategy> const &, size_t i, size_t j, size_t k) {
            return StorageInfo{i, j, k};
        }
    } // namespace tmp_storage
} // namespace gridtools

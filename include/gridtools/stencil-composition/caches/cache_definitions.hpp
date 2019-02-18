/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
/**
   @file
   @brief File containing all definitions and enums required by the cache implementations
*/

namespace gridtools {
    /**
     * @enum cache_io_policy
     * Enum listing the cache IO policies
     */
    enum class cache_io_policy {
        fill_and_flush, /**< Read values from the cached field and write the result back */
        fill,           /**< Read values form the cached field but do not write back */
        flush,          /**< Write values back the the cached field but do not read in */
        local           /**< Local only cache, neither read nor write the the cached field */
    };

    /**
     * @enum cache_type
     * enum with the different types of cache available
     */
    enum class cache_type {
        ij, // ij caches require synchronization capabilities, as different (i,j) grid points are
            // processed by parallel cores. GPU backend keeps them in shared memory
        k // processing of all the k elements is done by same thread, so resources for k caches can be private
          // and do not require synchronization. GPU backend uses registers.
    };
} // namespace gridtools

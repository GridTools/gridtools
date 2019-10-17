/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
/**
   @file
   @brief File containing metafunctions and traits common to all cache classes
*/

#pragma once

#include "../../meta.hpp"

namespace gridtools {
    template <class...>
    struct cache_map {};

    template <class Plh, class CacheTypes = meta::list<>, class CacheIOPolicies = meta::list<>>
    struct cache_info {
        using plh_t = Plh;
        using cache_types_t = CacheTypes;
        using cache_io_policies_t = CacheIOPolicies;
    };
} // namespace gridtools

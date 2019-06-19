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
#include "./cache.hpp"
#include "./cache_traits.hpp"

namespace gridtools {
    template <class Caches>
    using ij_caches = meta::filter<is_ij_cache, Caches>;

    template <class Caches>
    using ij_cache_args = meta::transform<cache_parameter, ij_caches<Caches>>;

    template <class Caches>
    using k_caches = meta::filter<is_k_cache, Caches>;

    template <class Caches>
    using k_cache_args = meta::transform<cache_parameter, k_caches<Caches>>;
} // namespace gridtools

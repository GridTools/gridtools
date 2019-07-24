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
   @brief File containing the definition of caches. They are the API exposed to the user to describe
   parameters that will be cached in a on-chip memory.
*/

#pragma once

#include <tuple>
#include <type_traits>

#include "../../common/defs.hpp"
#include "../../meta/type_traits.hpp"
#include "../arg.hpp"
#include "../interval.hpp"
#include "../location_type.hpp"
#include "./cache_definitions.hpp"

namespace gridtools {

    namespace detail {
        /**
         * @struct cache_impl
         * The cache type is described with a template parameter to the class
         * Caching assumes a parallelization model where all the processing all elements in the vertical dimension are
         * private to each parallel thread,
         * while the processing of grid points in the horizontal plane is executed by different parallel threads.
         * Those caches that cover data in the horizontal (IJ) are accessed by parallel core units, and
         * therefore require synchronization capabilities (for example shared memory in the GPU), like IJ caches.
         * On the contrary caches in the K dimension are only accessed by one thread, and therefore resources can be
         * allocated in on-chip without synchronization capabilities (for example registers in GPU)
         * @tparam  cacheType type of cache
         * @tparam Arg argument with parameter being cached
         * @tparam CacheIOPolicy IO policy for cache
         */
        template <cache_type CacheType, class Arg, cache_io_policy CacheIOPolicy>
        struct cache_impl {
            GT_STATIC_ASSERT(is_plh<Arg>::value, GT_INTERNAL_ERROR);
            using arg_t = Arg;
            static constexpr cache_type cacheType = CacheType;
            static constexpr cache_io_policy cacheIOPolicy = CacheIOPolicy;
        };
    } // namespace detail

    template <typename T>
    struct is_cache : std::false_type {};

    template <cache_type cacheType, class Arg, cache_io_policy cacheIOPolicy>
    struct is_cache<detail::cache_impl<cacheType, Arg, cacheIOPolicy>> : std::true_type {};

    /**
     *	@brief function that forms a vector of caches that share the same cache type and input/output policy
     *	@tparam cacheType type of cache (e.g., IJ, IJK, ...)
     *	@tparam cacheIOPolicy input/output policy (e.g., cFill, cLocal, ...)
     *	@tparam Args arbitrary number of storages that should be cached
     *	@return tuple of caches
     */
    template <cache_type cacheType, cache_io_policy cacheIOPolicy, class... Args>
    std::tuple<detail::cache_impl<cacheType, Args, cacheIOPolicy>...> cache(Args...) {
        GT_STATIC_ASSERT(sizeof...(Args) > 0, "Cannot build cache sequence without argument");
        GT_STATIC_ASSERT(
            conjunction<is_plh<Args>...>::value, "argument passed to cache is not of the right arg<> type");
        // TODO ICO_STORAGE
#ifdef GT_ICOSAHEDRAL_GRIDS
        GT_STATIC_ASSERT(
            (!disjunction<std::is_same<typename Args::location_t, enumtype::default_location_type>...>::value),
            "args in irregular grids require a location type");
#endif
        return {};
    }
} // namespace gridtools

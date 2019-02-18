/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
/**
   @file
   @brief File containing metafunctions and traits common to all cache classes
*/

#pragma once

#include <type_traits>

#include "../../meta/macros.hpp"
#include "cache.hpp"

namespace gridtools {

    /**
     * @struct is_ij_cache
     * metafunction determining if a type is a cache of IJ type
     */
    template <typename T>
    struct is_ij_cache : std::false_type {};

    template <typename Arg, cache_io_policy cacheIOPolicy>
    struct is_ij_cache<detail::cache_impl<cache_type::ij, Arg, cacheIOPolicy>> : std::true_type {};

    /**
     * @struct is_k_cache
     * metafunction determining if a type is a cache of K type
     */
    template <typename T>
    struct is_k_cache : std::false_type {};

    template <typename Arg, cache_io_policy cacheIOPolicy>
    struct is_k_cache<detail::cache_impl<cache_type::k, Arg, cacheIOPolicy>> : std::true_type {};

    /**
     * @struct is_flushing_cache
     * metafunction determining if a type is a flush cache
     */
    template <typename T>
    struct is_flushing_cache : std::false_type {};

    template <cache_type cacheType, typename Arg>
    struct is_flushing_cache<detail::cache_impl<cacheType, Arg, cache_io_policy::flush>> : std::true_type {};

    template <cache_type cacheType, typename Arg>
    struct is_flushing_cache<detail::cache_impl<cacheType, Arg, cache_io_policy::fill_and_flush>> : std::true_type {};

    /**
     * @struct is_filling_cache
     * metafunction determining if a type is a filling cache
     */
    template <typename T>
    struct is_filling_cache : std::false_type {};

    template <cache_type cacheType, typename Arg>
    struct is_filling_cache<detail::cache_impl<cacheType, Arg, cache_io_policy::fill>> : std::true_type {};

    template <cache_type cacheType, typename Arg>
    struct is_filling_cache<detail::cache_impl<cacheType, Arg, cache_io_policy::fill_and_flush>> : std::true_type {};

    template <typename T>
    struct is_local_cache : std::false_type {};

    template <cache_type cacheType, typename Arg>
    struct is_local_cache<detail::cache_impl<cacheType, Arg, cache_io_policy::local>> : std::true_type {};

    /**
     * @struct cache_parameter
     *  trait returning the parameter Arg type of a user provided cache
     */

    GT_META_LAZY_NAMESPACE {
        template <typename T>
        struct cache_parameter;

        template <cache_type cacheType, typename Arg, cache_io_policy cacheIOPolicy>
        struct cache_parameter<detail::cache_impl<cacheType, Arg, cacheIOPolicy>> {
            using type = Arg;
        };
    }
    GT_META_DELEGATE_TO_LAZY(cache_parameter, typename T, T);

} // namespace gridtools

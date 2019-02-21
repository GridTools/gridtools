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

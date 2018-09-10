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
   @brief File containing the definition of caches. They are the API exposed to the user to describe
   parameters that will be cached in a on-chip memory.
*/

#pragma once

#include <tuple>
#include <type_traits>

#include <boost/fusion/include/mpl.hpp>
#include <boost/fusion/include/std_tuple.hpp>

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/meta.hpp"
#include "../../common/generic_metafunctions/type_traits.hpp"
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
         * Those caches that cover data in the horizontal (IJ and IJK) are accessed by parallel core units, and
         * therefore require synchronization capabilities (for example shared memory in the GPU), like IJ or IJK caches.
         * On the contrary caches in the K dimension are only accessed by one thread, and therefore resources can be
         * allocated
         * in on-chip without synchronization capabilities (for example registers in GPU)
         * @tparam  cacheType type of cache
         * @tparam Arg argument with parameter being cached
         * @tparam CacheIOPolicy IO policy for cache
         * @tparam Interval vertical interval of validity of the cache
         */
        template <cache_type CacheType, class Arg, cache_io_policy cacheIOPolicy, class Interval>
        struct cache_impl {
            GRIDTOOLS_STATIC_ASSERT(is_arg<Arg>::value, GT_INTERNAL_ERROR);
            using arg_t = Arg;
            using interval_t = Interval;
            static constexpr cache_type cacheType = CacheType;
            static constexpr cache_io_policy ccacheIOPolicy = cacheIOPolicy;
        };
    } // namespace detail

    /**
     *	@brief function that forms a vector of caches that share the same cache type and input/output policy
     *	@tparam cacheType type of cache (e.g., IJ, IJK, ...)
     *	@tparam cacheIOPolicy input/output policy (e.g., cFill, cLocal, ...)
     *	@tparam Args arbitrary number of storages that should be cached
     *	@return tuple of caches
     */
    template <cache_type cacheType, cache_io_policy cacheIOPolicy, class Interval = void, class... Args>
    std::tuple<detail::cache_impl<cacheType, Args, cacheIOPolicy, Interval>...> cache(Args...) {
        GRIDTOOLS_STATIC_ASSERT(sizeof...(Args) > 0, "Cannot build cache sequence without argument");
        GRIDTOOLS_STATIC_ASSERT(
            conjunction<is_arg<Args>...>::value, "argument passed to cache is not of the right arg<> type");
        // TODO ICO_STORAGE
#ifndef STRUCTURED_GRIDS
        GRIDTOOLS_STATIC_ASSERT(
            (!disjunction<std::is_same<typename Args::location_t, enumtype::default_location_type>...>::value),
            "args in irregular grids require a location type");
#endif
        GRIDTOOLS_STATIC_ASSERT(std::is_void<Interval>::value || cacheType == K,
            "Passing an interval to the cache<> construct is only allowed and required by the K caches");
        GRIDTOOLS_STATIC_ASSERT(
            !std::is_void<Interval>::value || cacheType != K || cacheIOPolicy == cache_io_policy::local,
            "cache<K, ... > construct requires an interval (unless the IO policy is local)");
        GRIDTOOLS_STATIC_ASSERT(std::is_void<Interval>::value || is_interval<Interval>::value,
            "Invalid Interval type passed to cache construct");

        return {};
    }
} // namespace gridtools

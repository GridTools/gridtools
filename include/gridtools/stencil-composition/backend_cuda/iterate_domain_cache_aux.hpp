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
/*
 * @file
 * @brief file containing helper infrastructure, functors and metafunctions
 *  for the cache functionality of the iterate domain.
 */

#pragma once

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../meta/macros.hpp"
#include "../../meta/make_indices.hpp"
#include "../accessor.hpp"
#include "../caches/cache_definitions.hpp"
#include "../grid.hpp"

namespace gridtools {

    namespace _impl {

        /**
         * @brief Performs cache fill and flush operations from and to main memory.
         * @tparam CacheIOPolicy fill or flush
         * @tparam AccIndex accessor index
         * @tparam BaseOffset base offset along k-axis
         * @tparam IterateDomain iterate domain to access main device memory
         * @tparam CacheStorage cache storage to use
         */
        template <cache_io_policy CacheIOPolicy,
            typename AccIndex,
            int_t BaseOffset,
            typename IterateDomain,
            typename CacheStorage>
        struct io_operator {
            GT_FUNCTION io_operator(IterateDomain const &it_domain, CacheStorage &cache_storage)
                : m_it_domain(it_domain), m_cache_storage(cache_storage) {}

            template <typename Offset>
            GT_FUNCTION void operator()(Offset) const {
                static constexpr int_t offset = BaseOffset + (int_t)Offset::value;

                using acc_t = accessor<AccIndex::value, intent::inout, extent<0, 0, 0, 0, offset, offset>>;
                static constexpr acc_t acc(0, 0, offset);

                // perform an out-of-bounds check
                if (auto mem_ptr =
                        m_it_domain.template get_gmem_ptr_in_bounds<AccIndex, BaseOffset + (int_t)Offset::value>()) {
                    // fill or flush cache
                    if (CacheIOPolicy == cache_io_policy::fill)
                        m_cache_storage.at(acc) = *mem_ptr;
                    else
                        *mem_ptr = m_cache_storage.at(acc);
                }
            }

            IterateDomain const &m_it_domain;
            CacheStorage &m_cache_storage;
        };

        template <cache_io_policy CacheIOPolicy,
            typename AccIndex,
            int_t BaseOffset,
            typename IterateDomain,
            typename CacheStorage>
        GT_FUNCTION io_operator<CacheIOPolicy, AccIndex, BaseOffset, IterateDomain, CacheStorage> make_io_operator(
            IterateDomain const &it_domain, CacheStorage &cache_storage) {
            return {it_domain, cache_storage};
        }

        /**
         * @brief Base class for functors that sync k-caches with device memory.
         * @tparam KCachesTuple fusion tuple as map of pairs of <index,cache_storage>
         * @tparam KCachesMap mpl map of <index, cache_storage>
         * @tparam IterateDomain is the iterate domain
         * @tparam IterationPolicy: forward, backward
         * @tparam CacheIOPolicy the cache io policy: fill, flush
         */
        template <typename KCachesTuple,
            typename KCachesMap,
            typename IterateDomain,
            typename IterationPolicy,
            cache_io_policy CacheIOPolicy>
        struct io_cache_functor_base {
          private:
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_iterate_domain<IterateDomain>::value), GT_INTERNAL_ERROR);

            IterateDomain const &m_it_domain;
            KCachesTuple &m_kcaches;

            // shortcurts for forward backward iteration policy
            static constexpr bool forward = IterationPolicy::value == execution::forward;
            static constexpr bool backward = IterationPolicy::value == execution::backward;

            // shortcuts for fill and flush io policy
            static constexpr bool fill = CacheIOPolicy == cache_io_policy::fill;
            static constexpr bool flush = CacheIOPolicy == cache_io_policy::flush;

          protected:
            // `tail` is true if we have to fill or flush the tail (kminus side) of the cache, false if we have to
            // fill or flush the head (kplus side) of the cache.
            static constexpr bool tail = (backward && fill) || (forward && flush);

            /**
             * @brief Syncs the elements of k-cache with index `Idx` for all offsets in range [`SyncStart`, `SyncEnd`].
             */
            template <typename Idx, int_t SyncStart, int_t SyncEnd = SyncStart>
            GT_FUNCTION void sync() const {
                GRIDTOOLS_STATIC_ASSERT(forward || backward, "k-caches only support forward and backward iteration");
                GRIDTOOLS_STATIC_ASSERT(fill || flush, "io policy must be either fill or flush");
                static constexpr uint_t sync_size = (uint_t)(SyncEnd - SyncStart + 1);
                using range = GT_META_CALL(meta::make_indices_c, sync_size);
                auto &cache_st = boost::fusion::at_key<Idx>(m_kcaches);
                host_device::for_each<range>(make_io_operator<CacheIOPolicy, Idx, SyncStart>(m_it_domain, cache_st));
            }

          public:
            GT_FUNCTION
            io_cache_functor_base(IterateDomain const &it_domain, KCachesTuple &kcaches)
                : m_it_domain(it_domain), m_kcaches(kcaches) {}
        };

        /**
         * @brief Functor that syncs k-caches with main memory, this is the implementation for fill and flush caches
         * used on all k-levels of the iteration.
         * @tparam KCachesTuple fusion tuple as map of pairs of <index,cache_storage>
         * @tparam KCachesMap mpl map of <index, cache_storage>
         * @tparam IterateDomain is the iterate domain
         * @tparam IterationPolicy: forward, backward
         * @tparam CacheIOPolicy the cache io policy: fill, flush
         */
        template <typename KCachesTuple,
            typename KCachesMap,
            typename IterateDomain,
            typename IterationPolicy,
            cache_io_policy CacheIOPolicy>
        struct io_cache_functor
            : io_cache_functor_base<KCachesTuple, KCachesMap, IterateDomain, IterationPolicy, CacheIOPolicy> {
            using base = io_cache_functor_base<KCachesTuple, KCachesMap, IterateDomain, IterationPolicy, CacheIOPolicy>;
            using base::io_cache_functor_base;

            /**
             * @brief Syncs one level of the cache with main memory.
             */
            template <typename Idx>
            GT_FUNCTION void operator()(Idx) const {
                using kcache_storage_t = typename boost::mpl::at<KCachesMap, Idx>::type;

                // lowest and highest index in cache storage
                static constexpr int_t kminus = kcache_storage_t::kminus_t::value;
                static constexpr int_t kplus = kcache_storage_t::kplus_t::value;

                // cache index at which we need to sync (single element)
                static constexpr int_t sync_point = base::tail ? kminus : kplus;

                base::template sync<Idx, sync_point>();
            }
        };

        /**
         * @brief Functor that syncs k-caches with main memory, this is the implementation for fill and flush caches
         * used on the beginning and end levels of the iteration.
         * @tparam KCachesTuple fusion tuple as map of pairs of <index,cache_storage>
         * @tparam KCachesMap mpl map of <index, cache_storage>
         * @tparam IterateDomain is the iterate domain
         * @tparam IterationPolicy: forward, backward
         * @tparam CacheIOPolicy the cache io policy: fill, flush
         */
        template <typename KCachesTuple,
            typename KCachesMap,
            typename IterateDomain,
            typename IterationPolicy,
            cache_io_policy CacheIOPolicy>
        struct endpoint_io_cache_functor
            : io_cache_functor_base<KCachesTuple, KCachesMap, IterateDomain, IterationPolicy, CacheIOPolicy> {
            using base = io_cache_functor_base<KCachesTuple, KCachesMap, IterateDomain, IterationPolicy, CacheIOPolicy>;
            using base::io_cache_functor_base;

            /**
             * @brief Sync implementation for non-endpoint-only caches (i.e. fill, flush).
             */
            template <typename Idx>
            GT_FUNCTION void operator()(Idx) const {
                using kcache_storage_t = typename boost::mpl::at<KCachesMap, Idx>::type;

                // lowest and highest index in cache storage
                static constexpr int_t kminus = kcache_storage_t::kminus_t::value;
                static constexpr int_t kplus = kcache_storage_t::kplus_t::value;

                // with fill or flush caches, we need to load/store one element less at the begin and endpoints as the
                // non-endpoint fill or flush on the same k-level will handle this already
                static constexpr int_t kminus_offset = base::tail ? 1 : 0;
                static constexpr int_t kplus_offset = !base::tail ? -1 : 0;

                // choose lower and upper cache index for syncing
                static constexpr int_t sync_start = kminus + kminus_offset;
                static constexpr int_t sync_end = kplus + kplus_offset;

                base::template sync<Idx, sync_start, sync_end>();
            }
        };
    } // namespace _impl
} // namespace gridtools

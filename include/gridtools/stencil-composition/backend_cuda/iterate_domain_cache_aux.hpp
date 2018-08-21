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
#include "../../common/generic_metafunctions/meta.hpp"
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
                constexpr int_t offset = BaseOffset + (int_t)Offset::value;

                using acc_t = accessor<AccIndex::value, enumtype::inout, extent<0, 0, 0, 0, offset, offset>>;
                constexpr acc_t acc(0, 0, offset);

                // perform an out-of-bounds check
                if (auto mem_ptr = m_it_domain.get_gmem_ptr_in_bounds(acc)) {
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
         * @struct io_cache_functor
         * functor that performs the io cache operations (fill and flush) from main memory into a kcache and viceversa
         * @tparam KCachesTuple fusion tuple as map of pairs of <index,cache_storage>
         * @tparam KCachesMap mpl map of <index, cache_storage>
         * @tparam IterateDomain is the iterate domain
         * @tparam IterationPolicy: forward, backward
         * @tparam Grid grid type
         * @tparam CacheIOPolicy the cache io policy: fill, flush
         * @tparam AtBeginOrEndPoint true if the performed operation is performed at a cache begin- or endpoint
         */
        template <typename KCachesTuple,
            typename KCachesMap,
            typename IterateDomain,
            typename IterationPolicy,
            cache_io_policy CacheIOPolicy>
        struct io_cache_functor_base {
          protected:
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_iterate_domain<IterateDomain>::value), GT_INTERNAL_ERROR);

            IterateDomain const &m_it_domain;
            KCachesTuple &m_kcaches;

            // shortcurts for forward backward iteration policy
            static constexpr bool forward = IterationPolicy::value == enumtype::forward;
            static constexpr bool backward = !forward;

            // shortcuts for fill and flush io policy
            static constexpr bool fill = CacheIOPolicy == cache_io_policy::fill;
            static constexpr bool flush = !fill;

            // `tail` is true if we have to fill or flush the tail (kminus side) of the cache, false if we have to
            // fill or flush the head (kplus side) of the cache.
            static constexpr bool tail = (backward && fill) || (forward && flush);

            template <typename Idx, int_t SyncStart, int_t SyncEnd = SyncStart>
            GT_FUNCTION void sync() const {
                GRIDTOOLS_STATIC_ASSERT(
                    IterationPolicy::value == enumtype::forward || IterationPolicy::value == enumtype::backward,
                    "k-caches only support forward and backward iteration");
                GRIDTOOLS_STATIC_ASSERT(
                    CacheIOPolicy == cache_io_policy::fill || CacheIOPolicy == cache_io_policy::flush,
                    "io policy must be either fill or flush");
                constexpr uint_t sync_size = (uint_t)(SyncEnd - SyncStart + 1);
                using range = GT_META_CALL(meta::make_indices_c, sync_size);
                auto &cache_st = boost::fusion::at_key<Idx>(m_kcaches);
                gridtools::for_each<range>(make_io_operator<CacheIOPolicy, Idx, SyncStart>(m_it_domain, cache_st));
            }

          public:
            GT_FUNCTION
            io_cache_functor_base(IterateDomain const &it_domain, KCachesTuple &kcaches)
                : m_it_domain(it_domain), m_kcaches(kcaches) {}
        };

        template <typename KCachesTuple,
            typename KCachesMap,
            typename IterateDomain,
            typename IterationPolicy,
            cache_io_policy CacheIOPolicy>
        struct io_cache_functor
            : io_cache_functor_base<KCachesTuple, KCachesMap, IterateDomain, IterationPolicy, CacheIOPolicy> {
            using base = io_cache_functor_base<KCachesTuple, KCachesMap, IterateDomain, IterationPolicy, CacheIOPolicy>;
            using base::io_cache_functor_base;

            template <typename Idx>
            GT_FUNCTION void operator()(Idx) const {
                using kcache_storage_t = typename boost::mpl::at<KCachesMap, Idx>::type;

                // lowest index in cache storage
                constexpr int_t kminus = kcache_storage_t::kminus_t::value;
                // highest index in cache storage
                constexpr int_t kplus = kcache_storage_t::kplus_t::value;

                // cache index at which we need to sync (single element)
                constexpr int_t sync_point = base::tail ? kminus : kplus;

                base::template sync<Idx, sync_point>();
            }
        };

        template <typename KCachesTuple,
            typename KCachesMap,
            typename IterateDomain,
            typename IterationPolicy,
            cache_io_policy CacheIOPolicy>
        struct endpoint_io_cache_functor
            : io_cache_functor_base<KCachesTuple, KCachesMap, IterateDomain, IterationPolicy, CacheIOPolicy> {
            using base = io_cache_functor_base<KCachesTuple, KCachesMap, IterateDomain, IterationPolicy, CacheIOPolicy>;
            using base::io_cache_functor_base;

            template <typename Idx>
            GT_FUNCTION void operator()(Idx) const {
                using kcache_storage_t = typename boost::mpl::at<KCachesMap, Idx>::type;
                using kcache_t = typename kcache_storage_t::cache_t;

                // lowest and highest index in cache storage
                constexpr int_t kminus = kcache_storage_t::kminus_t::value;
                constexpr int_t kplus = kcache_storage_t::kplus_t::value;

                // true iff cache operations are only performed at the begin and endpoint
                constexpr bool endpoint_only = (kcache_t::ccacheIOPolicy == cache_io_policy::bpfill) ||
                                               (kcache_t::ccacheIOPolicy == cache_io_policy::epflush);

                // with fill or flush caches, we need to load/store one element less at the begin and endpoints as the
                // non-endpoint fill or flush on the same k-level will handle this already
                constexpr int_t kminus_offset = (base::tail && !endpoint_only) ? 1 : 0;
                constexpr int_t kplus_offset = (!base::tail && !endpoint_only) ? -1 : 0;

                // endpoint-only caches have a window, for others we use a zero-window for the limits
                using window_t =
                    typename std::conditional<endpoint_only, typename kcache_t::kwindow_t, window<0, 0>>::type;

                // choose lower and upper cache index for syncing
                constexpr int_t sync_start = base::tail ? kminus + kminus_offset : window_t::m_;
                constexpr int_t sync_end = base::tail ? window_t::p_ : kplus + kplus_offset;

                // endpoint flushes happen after the last slide, so we need to use add an additional offset in this case
                constexpr int_t flush_offset = base::fill ? 0 : base::forward ? -1 : 1;
                constexpr int_t shifted_sync_start = clamp_to_cache_krange<kcache_storage_t>(sync_start + flush_offset);
                constexpr int_t shifted_sync_end = clamp_to_cache_krange<kcache_storage_t>(sync_end + flush_offset);

                base::template sync<Idx, shifted_sync_start, shifted_sync_end>();
            }

            template <typename CacheStorage>
            GT_FUNCTION static constexpr int_t clamp_to_cache_krange(int_t index) {
                return index < CacheStorage::kminus_t::value
                           ? CacheStorage::kminus_t::value
                           : index > CacheStorage::kplus_t::value ? CacheStorage::kplus_t::value : index;
            }
        };
    } // namespace _impl
} // namespace gridtools

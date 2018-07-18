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

        template <typename IterateDomain, typename CacheStorage>
        struct fill_action {
            GT_FUNCTION fill_action(IterateDomain const &it_domain, CacheStorage &cache_storage)
                : m_it_domain(it_domain), m_cache_storage(cache_storage) {}

            template <typename Accessor>
            GT_FUNCTION void operator()(const Accessor &acc) const {
                m_cache_storage.at(acc) = m_it_domain.get_gmem_value(acc);
            }

            IterateDomain const &m_it_domain;
            CacheStorage &m_cache_storage;
        };

        template <typename IterateDomain, typename CacheStorage>
        struct flush_action {
            GT_FUNCTION flush_action(IterateDomain const &it_domain, CacheStorage const &cache_storage)
                : m_it_domain(it_domain), m_cache_storage(cache_storage) {}

            template <typename Accessor>
            GT_FUNCTION void operator()(const Accessor &acc) const {
                m_it_domain.get_gmem_value(acc) = m_cache_storage.at(acc);
            }

            IterateDomain const &m_it_domain;
            CacheStorage const &m_cache_storage;
        };

        template <cache_io_policy CacheIOPolicy, typename IterateDomain, typename CacheStorage>
        struct io_action_type;

        template <typename IterateDomain, typename CacheStorage>
        struct io_action_type<cache_io_policy::fill, IterateDomain, CacheStorage> {
            using type = fill_action<IterateDomain, CacheStorage>;
        };

        template <typename IterateDomain, typename CacheStorage>
        struct io_action_type<cache_io_policy::flush, IterateDomain, CacheStorage> {
            using type = flush_action<IterateDomain, CacheStorage>;
        };

        template <cache_io_policy CacheIOPolicy, typename IterateDomain, typename CacheStorage>
        GT_FUNCTION typename io_action_type<CacheIOPolicy, IterateDomain, CacheStorage>::type make_io_action(
            IterateDomain const &it_domain, CacheStorage &cache_storage) {
            using io_action_t = typename io_action_type<CacheIOPolicy, IterateDomain, CacheStorage>::type;
            return io_action_t(it_domain, cache_storage);
        }

        template <class CacheStorage>
        GT_FUNCTION constexpr int_t clamp_to_storage_krange(int_t index) {
            return index < CacheStorage::kminus_t::value
                       ? CacheStorage::kminus_t::value
                       : index > CacheStorage::kplus_t::value ? CacheStorage::kplus_t::value : index;
        }

        /**
         * @struct flush_mem_accessor
         * functor that will synchronize the last level of the cache ring-buffer with main memory, before iteration goes
         * to the next k-level
         * \tparam AccIndex index of the accessor
         * \tparam InitialOffset additional offset to be applied to the accessor
         */
        template <typename Action, typename AccIndex, int_t InitialOffset = 0>
        struct mem_access {
            GT_FUNCTION mem_access(Action const &action, int_t min_offset, int_t max_offset)
                : m_action(action), m_min_offset(min_offset), m_max_offset(max_offset) {}

            template <typename Offset>
            GT_FUNCTION void operator()(Offset) const {
                constexpr int_t offset = (int_t)Offset::value + InitialOffset;
                using acc_t = accessor<AccIndex::value, enumtype::inout, extent<0, 0, 0, 0, offset, offset>>;
                if (m_min_offset <= offset && offset <= m_max_offset) {
                    constexpr acc_t acc(0, 0, offset);
                    m_action(acc);
                }
            }

            Action const &m_action;
            const int_t m_min_offset, m_max_offset;
        };

        template <typename IterateDomain,
            typename CacheStorage,
            typename AccIndex,
            cache_io_policy CacheIOPolicy,
            int_t InitialOffset = 0>
        struct io_operator;

        template <typename IterateDomain, typename CacheStorage, typename AccIndex, int_t InitialOffset>
        struct io_operator<IterateDomain, CacheStorage, AccIndex, cache_io_policy::fill, InitialOffset> {
            using type = mem_access<fill_action<IterateDomain, CacheStorage>, AccIndex, InitialOffset>;
        };

        template <typename IterateDomain, typename CacheStorage, typename AccIndex, int_t InitialOffset>
        struct io_operator<IterateDomain, CacheStorage, AccIndex, cache_io_policy::flush, InitialOffset> {
            using type = mem_access<flush_action<IterateDomain, CacheStorage>, AccIndex, InitialOffset>;
        };

        enum class cache_section { head, tail };

        /**
         * compute the section of the kcache that is at the front according to the iteration policy
         */
        template <typename IterationPolicy>
        GT_FUNCTION constexpr cache_section compute_kcache_front(cache_io_policy cache_io_policy_) {
            return ((IterationPolicy::value == enumtype::backward) && (cache_io_policy_ == cache_io_policy::fill) ||
                       ((IterationPolicy::value == enumtype::forward) && (cache_io_policy_ == cache_io_policy::flush)))
                       ? cache_section::tail
                       : cache_section::head;
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
         */
        template <typename KCachesTuple,
            typename KCachesMap,
            typename IterateDomain,
            typename IterationPolicy,
            typename Grid,
            cache_io_policy CacheIOPolicy,
            bool Endpoint = false>
        struct io_cache_functor {
            GRIDTOOLS_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_iterate_domain<IterateDomain>::value), GT_INTERNAL_ERROR);

            GT_FUNCTION
            io_cache_functor(
                IterateDomain const &it_domain, KCachesTuple &kcaches, const int_t klevel, Grid const &grid)
                : m_it_domain(it_domain), m_kcaches(kcaches), m_klevel(klevel), m_grid(grid) {}

            IterateDomain const &m_it_domain;
            KCachesTuple &m_kcaches;
            const int_t m_klevel;
            Grid const &m_grid;

            template <typename Idx>
            GT_FUNCTION void operator()(Idx const &) const {
                using kcache_storage_t = typename boost::mpl::at<KCachesMap, Idx>::type;
                using kcache_t = typename kcache_storage_t::cache_t;
                using window_t = typename kcache_t::kwindow_t;
                GRIDTOOLS_STATIC_ASSERT(((IterationPolicy::value != enumtype::parallel) ||
                                            (kcache_t::ccacheIOPolicy != cache_io_policy::bpfill &&
                                                kcache_t::ccacheIOPolicy != cache_io_policy::epflush)),
                    "bpfill and epflush policies can not be used with a kparallel iteration strategy");

                constexpr bool endpoint_only = kcache_t::ccacheIOPolicy == cache_io_policy::bpfill ||
                                               kcache_t::ccacheIOPolicy == cache_io_policy::epflush;

                constexpr bool tail = compute_kcache_front<IterationPolicy>(CacheIOPolicy) == cache_section::tail;
                constexpr int_t endpoint_flush_offset = (CacheIOPolicy == cache_io_policy::fill)
                                                            ? 0
                                                            : (IterationPolicy::value == enumtype::forward) ? -1 : 1;

                const int_t from =
                    m_grid.template value_at<typename kcache_storage_t::cache_t::interval_t::FromLevel>();
                const int_t to = m_grid.template value_at<typename kcache_storage_t::cache_t::interval_t::ToLevel>();
                assert(from <= to);
                const int_t kplus_limit = endpoint_only ? window_t::p_ + endpoint_flush_offset : to - m_klevel;
                const int_t kminus_limit = endpoint_only ? window_t::m_ + endpoint_flush_offset : from - m_klevel;

                auto &cache_st = boost::fusion::at_key<Idx>(m_kcaches);
                auto action = make_io_action<CacheIOPolicy>(m_it_domain, cache_st);

                constexpr int_t kminus = kcache_storage_t::kminus_t::value;
                constexpr int_t kplus = kcache_storage_t::kplus_t::value;

                constexpr int_t endpoint_sync_start =
                    clamp_to_storage_krange<kcache_storage_t>((tail ? kminus : window_t::m_) + endpoint_flush_offset);
                constexpr int_t endpoint_sync_end =
                    clamp_to_storage_krange<kcache_storage_t>((tail ? window_t::p_ : kplus) + endpoint_flush_offset);

                constexpr int_t center_sync_point = tail ? kminus : kplus;

                constexpr int_t sync_start = Endpoint ? endpoint_sync_start : center_sync_point;
                constexpr int_t sync_end = Endpoint ? endpoint_sync_end : center_sync_point;

                constexpr uint_t sync_size = (uint_t)(sync_end - sync_start + 1);

                using range = GT_META_CALL(meta::make_indices_c, sync_size);
                using io_op_t =
                    typename io_operator<IterateDomain, kcache_storage_t, Idx, CacheIOPolicy, sync_start>::type;
                gridtools::for_each<range>(io_op_t(action, kminus_limit, kplus_limit));
            }
        };
    } // namespace _impl
} // namespace gridtools

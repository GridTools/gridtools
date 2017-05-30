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

namespace gridtools {

    namespace _impl {

        /**
         * @struct flush_mem_accessor
         * functor that will synchronize a cache with main memory
         * \tparam AccIndex index of the accessor
         * \tparam ExecutionPolicy : forward, backward
         * \tparam InitialOffset additional offset to be applied to the accessor
         */
        template < typename AccIndex, enumtype::execution ExecutionPolicy, int_t InitialOffset = 0 >
        struct flush_mem_accessor {
            /**
             * @struct apply struct of the functor
             * \tparam Offset integer that specifies the vertical offset of the cache parameter being synchronized
             */
            template < int_t Offset >
            struct apply_t {
                /**
                 * @brief apply the functor
                 * @param it_domain iterate domain
                 * @param cache_st cache storage
                 */
                template < typename IterateDomain, typename CacheStorage >
                GT_FUNCTION static int_t apply(IterateDomain const &it_domain, CacheStorage const &cache_st) {

                    typedef accessor< AccIndex::value,
                        enumtype::inout,
                        extent< 0, 0, 0, 0, -(Offset + InitialOffset), (Offset + InitialOffset) > > acc_t;
                    constexpr acc_t acc_(0,
                        0,
                        (ExecutionPolicy == enumtype::forward) ? -(Offset + InitialOffset) : (Offset + InitialOffset));
                    it_domain.gmem_access(acc_) = cache_st.at(acc_);
                    return 0;
                }
            };
        };

        /**
         * @struct fill_mem_accessor
         * functor that prefill a kcache (before starting the vertical iteration) with initial values from main memory
         * \tparam AccIndex index of the accessor
         * \tparam ExecutionPolicy : forward, backward
         * \tparam InitialOffset additional offset to be applied to the accessor
         */
        template < typename AccIndex, enumtype::execution ExecutionPolicy, int_t InitialOffset = 0 >
        struct fill_mem_accessor {
            /**
             * @struct apply struct of the functor
             * \tparam Offset integer that specifies the vertical offset of the cache parameter being synchronized
             */
            template < int_t Offset >
            struct apply_t {
                /**
                 * @brief apply the functor
                 * @param it_domain iterate domain
                 * @param cache_st cache storage
                 */
                template < typename IterateDomain, typename CacheStorage >
                GT_FUNCTION static int_t apply(IterateDomain const &it_domain, CacheStorage &cache_st) {

                    typedef accessor< AccIndex::value,
                        enumtype::in,
                        extent< 0, 0, 0, 0, -(Offset + InitialOffset), (Offset + InitialOffset) > > acc_t;
                    constexpr acc_t acc_(0,
                        0,
                        (ExecutionPolicy == enumtype::backward) ? -(Offset + InitialOffset) : (Offset + InitialOffset));
                    cache_st.at(acc_) = it_domain.gmem_access(acc_);
                    return 0;
                }
            };
        };

        template < typename AccIndex,
            enumtype::execution ExecutionPolicy,
            cache_io_policy CacheIOPolicy,
            int_t InitialOffset = 0 >
        struct io_operator;

        template < typename AccIndex, enumtype::execution ExecutionPolicy, int_t InitialOffset >
        struct io_operator< AccIndex, ExecutionPolicy, cache_io_policy::fill, InitialOffset > {
            using type = fill_mem_accessor< AccIndex, ExecutionPolicy, InitialOffset >;
        };

        template < typename AccIndex, enumtype::execution ExecutionPolicy, int_t InitialOffset >
        struct io_operator< AccIndex, ExecutionPolicy, cache_io_policy::flush, InitialOffset > {
            using type = flush_mem_accessor< AccIndex, ExecutionPolicy, InitialOffset >;
        };

        enum class cache_section { head, tail };

        /**
         * compute the section of the kcache that is at the front according to the iteration policy
         */
        template < typename IterationPolicy >
        GT_FUNCTION constexpr cache_section compute_kcache_front(cache_io_policy cache_io_policy_) {
            return ((IterationPolicy::value == enumtype::backward) && (cache_io_policy_ == cache_io_policy::fill) ||
                       ((IterationPolicy::value == enumtype::forward) && (cache_io_policy_ == cache_io_policy::flush)))
                       ? cache_section::tail
                       : cache_section::head;
        }

        /**
         * compute the maximum length of the section of the kcache that will be sync with main memory according to the
         * iteration policy
         */
        template < typename IterationPolicy, typename CacheStorage >
        GT_FUNCTION constexpr uint_t compute_section_kcache_to_sync_with_mem(cache_io_policy cache_io_policy_) {
            GRIDTOOLS_STATIC_ASSERT((boost::mpl::at_c< typename CacheStorage::minus_t::type, 2 >::type::value <= 0 &&
                                        boost::mpl::at_c< typename CacheStorage::plus_t::type, 2 >::type::value >= 0),
                GT_INTERNAL_ERROR);

            return (compute_kcache_front< IterationPolicy >(cache_io_policy_) == cache_section::tail)
                       ? (uint_t)-boost::mpl::at_c< typename CacheStorage::minus_t::type, 2 >::type::value
                       : (uint_t)boost::mpl::at_c< typename CacheStorage::plus_t::type, 2 >::type::value;
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
        template < typename KCachesTuple,
            typename KCachesMap,
            typename IterateDomain,
            typename IterationPolicy,
            typename Grid,
            cache_io_policy CacheIOPolicy >
        struct io_cache_functor {
            GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy< IterationPolicy >::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_iterate_domain< IterateDomain >::value), GT_INTERNAL_ERROR);

            GT_FUNCTION
            io_cache_functor(
                IterateDomain const &it_domain, KCachesTuple &kcaches, const int_t klevel, Grid const &grid)
                : m_it_domain(it_domain), m_kcaches(kcaches), m_klevel(klevel), m_grid(grid) {}

            IterateDomain const &m_it_domain;
            KCachesTuple &m_kcaches;
            const int_t m_klevel;
            Grid const &m_grid;

            template < typename Idx >
            GT_FUNCTION void operator()(Idx const &) const {
                typedef typename boost::mpl::at< KCachesMap, Idx >::type k_cache_storage_t;

                // compute the offset values that we will fill/flush from/to memory
                constexpr uint_t koffset =
                    compute_section_kcache_to_sync_with_mem< IterationPolicy, k_cache_storage_t >(CacheIOPolicy);

                // compute the limit level of the iteration space in k, below which we can not fill (in case of fill)
                // or beyond which we can not flush (in case of flush) since it might
                // produce an out of bounds when accessing main memory. This limit level is defined by the interval
                // associated to the kcache
                const int_t limit_lev =
                    (compute_kcache_front< IterationPolicy >(CacheIOPolicy) == cache_section::tail)
                        ? m_klevel -
                              m_grid.template value_at< typename k_cache_storage_t::cache_t::interval_t::FromLevel >()
                        : m_grid.template value_at< typename k_cache_storage_t::cache_t::interval_t::ToLevel >() -
                              m_klevel;

                if (koffset <= limit_lev) {
                    using io_op_t = typename io_operator< Idx, IterationPolicy::value, CacheIOPolicy >::type;

                    io_op_t::template apply_t< koffset >::apply(m_it_domain, boost::fusion::at_key< Idx >(m_kcaches));
                }
            }
        };

        /**
         * @struct endpoint_io_cache_functor
         * functor that performs the final flush or begin fill operation between main memory and a kcache, that is
         * executed
         * at the end of the vertical iteration(flush) or at the beginning the of iteration of the vertical interval
         * (fill)
         * @tparam KCachesTuple fusion tuple as map of pairs of <index,cache_storage>
         * @tparam KCachesMap mpl map of <index, cache_storage>
         * @tparam IterateDomain is the iterate domain
         * @tparam IterationPolicy: forward, backward
         * @tparam CacheIOPolicy cache io policy: fill or flush
         */
        template < typename KCachesTuple,
            typename KCachesMap,
            typename IterateDomain,
            typename IterationPolicy,
            cache_io_policy CacheIOPolicy >
        struct endpoint_io_cache_functor {

            GT_FUNCTION
            endpoint_io_cache_functor(IterateDomain const &it_domain, KCachesTuple &kcaches)
                : m_it_domain(it_domain), m_kcaches(kcaches) {}

            IterateDomain const &m_it_domain;
            KCachesTuple &m_kcaches;

            template < typename Idx >
            GT_FUNCTION void operator()(Idx const &) const {
                typedef typename boost::mpl::at< KCachesMap, Idx >::type k_cache_storage_t;
                using kcache_t = typename k_cache_storage_t::cache_t;

                // compute the maximum offset of all levels that we need to prefill or final flush
                constexpr uint_t koffset =
                    compute_section_kcache_to_sync_with_mem< IterationPolicy, k_cache_storage_t >(CacheIOPolicy);

                // compute the sequence of all offsets that we need to prefill or final flush
                using seq = gridtools::apply_gt_integer_sequence< typename gridtools::make_gt_integer_sequence<
                    int_t,
                    (kcache_t::ccacheIOPolicy == cache_io_policy::bpfill ||
                        kcache_t::ccacheIOPolicy == cache_io_policy::epflush)
                        ? koffset + 1
                        : koffset >::type >;
                using io_op_t = typename io_operator< Idx,
                    IterationPolicy::value,
                    CacheIOPolicy,
                    (kcache_t::ccacheIOPolicy == cache_io_policy::flush) ? (int_t)1 : (int_t)0 >::type;

                auto &cache_st = boost::fusion::at_key< Idx >(m_kcaches);
                seq::template apply_void_lambda< io_op_t::apply_t >(m_it_domain, cache_st);
            }
        };
    } // namespace _impl
} // namespace gridtools

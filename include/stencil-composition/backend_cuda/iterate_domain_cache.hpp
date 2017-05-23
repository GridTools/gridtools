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
 * @brief file containing infrastructure to provide cache functionality to the iterate domain.
 * All caching operations performance by the iterate domain are delegated to the code written here
 *
 */

#pragma once

#include "../caches/cache_metafunctions.hpp"
#include "../caches/extract_extent_caches.hpp"
#include "common/defs.hpp"
#include "common/generic_metafunctions/fusion_map_to_mpl_map.hpp"
#include "common/generic_metafunctions/vector_to_map.hpp"
#include "stencil-composition/iterate_domain_fwd.hpp"
#include <boost/fusion/container/map/convert.hpp>
#include <boost/fusion/include/as_map.hpp>
#include <boost/fusion/include/pair.hpp>
#include <boost/fusion/support/pair.hpp>
#include <boost/mpl/copy_if.hpp>
#include <boost/mpl/filter_view.hpp>
#include "common/generic_metafunctions/vector_to_map.hpp"
#include "common/generic_metafunctions/fusion_map_to_mpl_map.hpp"
#include "stencil-composition/iterate_domain_fwd.hpp"
#include "../caches/cache_metafunctions.hpp"
#include "../caches/extract_extent_caches.hpp"
#include "../accessor_fwd.hpp"
#include "../../common/generic_metafunctions/sequence_to_vector.hpp"

namespace gridtools {

    /**
     * @struct slide_cache_functor
     * functor used to apply the slide operation on all kcache arguments of the kcache tuple
     */
    template < typename IterationPolicy >
    struct slide_cache_functor {
      public:
        GT_FUNCTION
        slide_cache_functor() {}

        template < typename Arg >
        GT_FUNCTION void operator()(Arg &arg_) const {
            arg_.second.template slide< IterationPolicy >();
        }
    };

    /**
     * @struct filter_map_indexes
     * metafunction that returns a sequence of all the indexes of the pair elements
     * in the map that fulfil the predicate
     * \tparam Map is a map of <index, cache_storage>
     * \tparam Pred predicate used to filter the map elements
     */
    template < typename Map, template < typename > class Pred >
    struct filter_map_indexes {
        template < typename Pair >
        struct apply_pred {
            typedef typename Pred< typename Pair::second::cache_t >::type type;
        };
        typedef
            typename boost::mpl::fold< Map,
                boost::mpl::vector0<>,
                boost::mpl::if_< apply_pred< boost::mpl::_2 >,
                                           boost::mpl::push_back< boost::mpl::_1, boost::mpl::first< boost::mpl::_2 > >,
                                           boost::mpl::_1 > >::type type;
    };

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

    /**
     * @class iterate_domain_cache
     * class that provides all the caching functionality needed by the iterate domain.
     * It keeps in type information all the caches setup by the user and provides methods to access cache storage and
     * perform
     * all the caching operations, like filling, sliding or flushing.
     */
    template < typename IterateDomainArguments >
    class iterate_domain_cache {
        DISALLOW_COPY_AND_ASSIGN(iterate_domain_cache);

        GRIDTOOLS_STATIC_ASSERT((is_iterate_domain_arguments< IterateDomainArguments >::value), GT_INTERNAL_ERROR);
        typedef typename IterateDomainArguments::esf_sequence_t esf_sequence_t;
        typedef typename IterateDomainArguments::cache_sequence_t cache_sequence_t;

      private:
        // checks if an arg is used by any of the esfs within a sequence
        template < typename EsfSequence, typename Arg >
        struct is_arg_used_in_esf_sequence {
            typedef typename boost::mpl::fold< EsfSequence,
                boost::mpl::false_,
                boost::mpl::or_< boost::mpl::_1, boost::mpl::contains< esf_args< boost::mpl::_2 >, Arg > > >::type type;
        };

      public:
        GT_FUNCTION
        iterate_domain_cache() {}

        GT_FUNCTION
        ~iterate_domain_cache() {}

        // remove caches which are not used by the stencil stages
        typedef typename boost::mpl::copy_if< cache_sequence_t,
            is_arg_used_in_esf_sequence< esf_sequence_t, cache_parameter< boost::mpl::_ > > >::type caches_t;

        // extract a sequence of extents for each ij cache
        typedef typename extract_ij_extents_for_caches< IterateDomainArguments >::type ij_cache_extents_map_t;

        // extract a sequence of extents for each k cache
        typedef typename extract_k_extents_for_caches< IterateDomainArguments >::type k_cache_extents_map_t;
        // compute the fusion vector of pair<index_type, cache_storage> for ij caches
        typedef typename get_cache_storage_tuple< IJ,
            caches_t,
            ij_cache_extents_map_t,
            typename IterateDomainArguments::physical_domain_block_size_t,
            typename IterateDomainArguments::local_domain_t >::type ij_caches_vector_t;

        // compute the fusion vector of pair<index_type, cache_storage> for k caches
        typedef typename get_cache_storage_tuple< K,
            caches_t,
            k_cache_extents_map_t,
            typename IterateDomainArguments::physical_domain_block_size_t,
            typename IterateDomainArguments::local_domain_t >::type k_caches_vector_t;

        // extract a fusion map from the fusion vector of pairs for ij caches
        typedef typename boost::fusion::result_of::as_map< ij_caches_vector_t >::type ij_caches_tuple_t;

        // extract a fusion map from the fusion vector of pairs for k caches
        typedef typename boost::fusion::result_of::as_map< k_caches_vector_t >::type k_caches_tuple_t;

        // compute an mpl from the previous fusion vector, to be used for compile time meta operations
        typedef typename fusion_map_to_mpl_map< ij_caches_tuple_t >::type ij_caches_map_t;

        // compute an mpl from the previous fusion vector, to be used for compile time meta operations
        typedef typename fusion_map_to_mpl_map< k_caches_tuple_t >::type k_caches_map_t;

        // list of indexes of kcaches that require flushing operations
        typedef typename filter_map_indexes< k_caches_map_t, is_flushing_cache >::type k_flushing_caches_indexes_t;

        // list of indexes of kcaches that require end-point flushing operations
        typedef typename filter_map_indexes< k_caches_map_t, is_epflushing_cache >::type k_epflushing_caches_indexes_t;

        // list of indexes of kcaches that require filling operations
        typedef typename filter_map_indexes< k_caches_map_t, is_filling_cache >::type k_filling_caches_indexes_t;

        // list of indexes of kcaches that require begin-point filling operations
        typedef typename filter_map_indexes< k_caches_map_t, is_bpfilling_cache >::type k_bpfilling_caches_indexes_t;

        // set of "bypass" caches
        typedef
            typename get_cache_set_for_type< bypass, caches_t, typename IterateDomainArguments::local_domain_t >::type
                bypass_caches_set_t;

        // associative container with all caches
        typedef typename get_cache_set< caches_t, typename IterateDomainArguments::local_domain_t >::type all_caches_t;

        // returns a k cache from the tuple
        template < typename IndexType >
        GT_FUNCTION typename boost::mpl::at< k_caches_map_t, IndexType >::type &RESTRICT get_k_cache() {
            GRIDTOOLS_STATIC_ASSERT(
                (boost::mpl::has_key< k_caches_map_t, IndexType >::value), "Accessing a non registered cached");
            return boost::fusion::at_key< IndexType >(m_k_caches_tuple);
        }

        // slide all the k caches
        template < typename IterationPolicy >
        GT_FUNCTION void slide_caches() {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy< IterationPolicy >::value), "error");
            boost::fusion::for_each(m_k_caches_tuple, slide_cache_functor< IterationPolicy >());
        }

        /**
         * @struct io_cache_functor
         * functor that performs the io cache operations (fill and flush) from main memory into a kcache and viceversa
         * @tparam IterateDomain is the iterate domain
         * @tparam IterationPolicy: forward, backward
         * @tparam Grid grid type
         * @tparam CacheIOPolicy the cache io policy: fill, flush
         */
        template < typename IterateDomain, typename IterationPolicy, typename Grid, cache_io_policy CacheIOPolicy >
        struct io_cache_functor {
            GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), "error");
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy< IterationPolicy >::value), "error");
            GRIDTOOLS_STATIC_ASSERT((is_iterate_domain< IterateDomain >::value), "error");

            GT_FUNCTION
            io_cache_functor(
                IterateDomain const &it_domain, k_caches_tuple_t &kcaches, const int_t klevel, Grid const &grid)
                : m_it_domain(it_domain), m_kcaches(kcaches), m_klevel(klevel), m_grid(grid) {}

            IterateDomain const &m_it_domain;
            k_caches_tuple_t &m_kcaches;
            const int_t m_klevel;
            Grid const &m_grid;

            template < typename Idx >
            GT_FUNCTION void operator()(Idx const &) const {
                typedef typename boost::mpl::at< k_caches_map_t, Idx >::type k_cache_storage_t;

                // compute the offset values that we will fill/flush from/to memory
                constexpr int_t koffset =
                    ((IterationPolicy::value == enumtype::backward) && CacheIOPolicy == cache_io_policy::fill) ||
                            ((IterationPolicy::value == enumtype::forward) && CacheIOPolicy == cache_io_policy::flush)
                        ? boost::mpl::at_c< typename k_cache_storage_t::minus_t::type, 2 >::type::value
                        : boost::mpl::at_c< typename k_cache_storage_t::plus_t::type, 2 >::type::value;

                constexpr int_t koffset_abs = koffset > 0 ? koffset : -koffset;

                typedef accessor< Idx::value,
                    enumtype::in,
                    extent< 0, 0, 0, 0, (koffset < 0) ? koffset : -koffset, (koffset > 0) ? koffset : -koffset > >
                    acc_t;
                constexpr acc_t acc_(0, 0, koffset);

                // compute the limit level of the iteration space in k, below which we can not fill (in case of fill)
                // or beyond which we can not flush (in case of flush) since it might
                // produce an out of bounds when accessing main memory. This limit level is defined by the interval
                // associated to the kcache
                const int_t limit_lev =
                    (IterationPolicy::value == enumtype::backward && CacheIOPolicy == cache_io_policy::fill) ||
                            (IterationPolicy::value == enumtype::forward && CacheIOPolicy == cache_io_policy::flush)
                        ? m_klevel -
                              m_grid.template value_at< typename k_cache_storage_t::cache_t::interval_t::FromLevel >()
                        : m_grid.template value_at< typename k_cache_storage_t::cache_t::interval_t::ToLevel >() -
                              m_klevel;

                if (koffset_abs <= limit_lev) {
                    using io_op_t = typename io_operator< Idx, IterationPolicy::value, CacheIOPolicy >::type;

                    io_op_t::template apply_t< koffset_abs >::apply(
                        m_it_domain, boost::fusion::at_key< Idx >(m_kcaches));
                }
            }
        };

        /**
         * @struct endpoint_io_cache_functor
         * functor that performs the final flush or begin fill operation between main memory and a kcache, that is
         * executed
         * at the end of the vertical iteration(flush) or at the beginning the of iteration of the vertical interval
         * (fill),
         * @tparam IterateDomain is the iterate domain
         * @tparam IterationPolicy: forward, backward
         * @tparam CacheIOPolicy cache io policy: fill or flush
         */
        template < typename IterateDomain, typename IterationPolicy, cache_io_policy CacheIOPolicy >
        struct endpoint_io_cache_functor {

            GT_FUNCTION
            endpoint_io_cache_functor(IterateDomain const &it_domain, k_caches_tuple_t &kcaches)
                : m_it_domain(it_domain), m_kcaches(kcaches) {}

            IterateDomain const &m_it_domain;
            k_caches_tuple_t &m_kcaches;

            template < typename Idx >
            GT_FUNCTION void operator()(Idx const &) const {
                typedef typename boost::mpl::at< k_caches_map_t, Idx >::type k_cache_storage_t;
                using kcache_t = typename k_cache_storage_t::cache_t;

                GRIDTOOLS_STATIC_ASSERT(
                    (boost::mpl::at_c< typename k_cache_storage_t::minus_t::type, 2 >::type::value <= 0 &&
                        boost::mpl::at_c< typename k_cache_storage_t::plus_t::type, 2 >::type::value >= 0),
                    "Error");

                // compute the maximum offset of all levels that we need to prefill or final flush
                constexpr uint_t koffset =
                    (IterationPolicy::value == enumtype::forward && CacheIOPolicy == cache_io_policy::flush) ||
                            (IterationPolicy::value == enumtype::backward && CacheIOPolicy == cache_io_policy::fill)
                        ? -boost::mpl::at_c< typename k_cache_storage_t::minus_t::type, 2 >::type::value
                        : boost::mpl::at_c< typename k_cache_storage_t::plus_t::type, 2 >::type::value;
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
                    (CacheIOPolicy == cache_io_policy::flush) ? (int_t)1 : (int_t)0 >::type;

                auto &cache_st = boost::fusion::at_key< Idx >(m_kcaches);
                seq::template apply_void_lambda< io_op_t::apply_t >(m_it_domain, cache_st);
            }
        };

        /**
         * fill next k level from main memory for all k caches. The position of the kcache being filled
         * depends on the iteration policy
         * \tparam IterationPolicy forward: backward
         * \param it_domain an iterate domain
         * \param klevel current k level index
         * \param grid a grid with loop bounds information
         */
        template < typename IterationPolicy, typename IterateDomain, typename Grid >
        GT_FUNCTION void fill_caches(IterateDomain const &it_domain, const int_t klevel, const Grid &grid) {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy< IterationPolicy >::value), "error");
            GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), "error");

            boost::mpl::for_each< k_filling_caches_indexes_t >(
                io_cache_functor< IterateDomain, IterationPolicy, Grid, cache_io_policy::fill >(
                    it_domain, m_k_caches_tuple, klevel, grid));
        }

        /**
         * flush the last k level of the ring buffer into main memory. The position of the kcache being flushed
         * depends on the iteration policy
         * \tparam IterationPolicy forward: backward
         * \param it_domain an iterate domain
         * \param klevel current k level index
         * \param grid a grid with loop bounds information
         */
        template < typename IterationPolicy, typename IterateDomain, typename Grid >
        GT_FUNCTION void flush_caches(IterateDomain const &it_domain, const int_t klevel, Grid const &grid) {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy< IterationPolicy >::value), "error");
            GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), "error");

            boost::mpl::for_each< k_flushing_caches_indexes_t >(
                io_cache_functor< IterateDomain, IterationPolicy, Grid, cache_io_policy::flush >(
                    it_domain, m_k_caches_tuple, klevel, grid));
        }

        /**
         * @struct kcache_final_flush_indexes
         * metafunction that computes the list of indexes of all k caches that require a final flush
         */
        template < typename IterationPolicy >
        struct kcache_final_flush_indexes {
            template < typename CacheStorage >
            struct is_end_index {
                //                GRIDTOOLS_STATIC_ASSERT((is_cache_storage< CacheStorage >::value), "Internal Error");
                using cache_t = typename CacheStorage::cache_t;
                using to_index = typename level_to_index< typename IterationPolicy::to >::type;

                static constexpr bool value = (IterationPolicy::value == enumtype::forward)
                                                  ? (interval_to_index< typename cache_t::interval_t >::type::value ==
                                                        level_to_index< typename IterationPolicy::to >::type::value)
                                                  : (interval_from_index< typename cache_t::interval_t >::type::value ==
                                                        level_to_index< typename IterationPolicy::to >::type::value);
            };

            // determine indexes of all k caches that require flushing, whose associated interval ends with the interval
            // of the current iteration
            // policy.
            using interval_flushing_indexes_t = typename boost::mpl::filter_view< k_flushing_caches_indexes_t,
                is_end_index< boost::mpl::at< k_caches_map_t, boost::mpl::_ > > >::type;

            // same for those k caches that need an end-point flush. Determine among them, which ones have an interval
            // whose end
            // matches the current interval
            using interval_epflushing_indexes_t =
                typename sequence_to_vector< typename boost::mpl::filter_view< k_epflushing_caches_indexes_t,
                    is_end_index< boost::mpl::at< k_caches_map_t, boost::mpl::_ > > >::type >::type;

            using type =
                typename boost::mpl::copy< interval_flushing_indexes_t,
                    boost::mpl::inserter< interval_epflushing_indexes_t,
                                               boost::mpl::push_back< boost::mpl::_1, boost::mpl::_2 > > >::type;
        };

        /**
         * @struct kcache_begin_fill_indexes
         * metafunction that computes the list of indexes of all k caches that require a begin pre-fill of the cache
         */
        template < typename IterationPolicy >
        struct kcache_begin_fill_indexes {
            template < typename CacheStorage >
            struct is_end_index {
                using cache_t = typename CacheStorage::cache_t;
                using to_index = typename level_to_index< typename IterationPolicy::to >::type;

                static constexpr bool value = (IterationPolicy::value == enumtype::forward)
                                                  ? (interval_from_index< typename cache_t::interval_t >::type::value ==
                                                        level_to_index< typename IterationPolicy::from >::type::value)
                                                  : (interval_to_index< typename cache_t::interval_t >::type::value ==
                                                        level_to_index< typename IterationPolicy::from >::type::value);
            };

            // determine indexes of all k caches that require filling, whose associated interval starts with the
            // interval
            // of the current iteration policy.
            using interval_filling_indexes_t = typename boost::mpl::filter_view< k_filling_caches_indexes_t,
                is_end_index< boost::mpl::at< k_caches_map_t, boost::mpl::_ > > >::type;

            // same for those k cache that require a begin-point filling
            using interval_bpfilling_indexes_t =
                typename sequence_to_vector< typename boost::mpl::filter_view< k_bpfilling_caches_indexes_t,
                    is_end_index< boost::mpl::at< k_caches_map_t, boost::mpl::_ > > >::type >::type;

            using type =
                typename boost::mpl::copy< interval_filling_indexes_t,
                    boost::mpl::inserter< interval_bpfilling_indexes_t,
                                               boost::mpl::push_back< boost::mpl::_1, boost::mpl::_2 > > >::type;
        };

        /**
         * Initial fill of the of the kcaches. Before the iteration over k starts, we need to prefill the k level
         * of the cache with k > 0 (<0) for the forward (backward) iteration policy
         * \tparam IterationPolicy forward: backward
         * \param it_domain an iterate domain
         */
        template < typename IterationPolicy, typename IterateDomain >
        GT_FUNCTION void begin_fill(IterateDomain const &it_domain) {
            typedef typename kcache_begin_fill_indexes< IterationPolicy >::type k_begin_filling_caches_indexes_t;
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy< IterationPolicy >::value), "error");
            boost::mpl::for_each< k_begin_filling_caches_indexes_t >(
                endpoint_io_cache_functor< IterateDomain, IterationPolicy, cache_io_policy::fill >(
                    it_domain, m_k_caches_tuple));
        }

        /**
         * Final flush of the of the kcaches. After the iteration over k is done, we still need to flush the remaining
         * k levels of the cache with k > 0 (<0) for the backward (forward) iteration policy
         * \tparam IterationPolicy forward: backward
         * \param it_domain an iterate domain
         */
        template < typename IterationPolicy, typename IterateDomain >
        GT_FUNCTION void final_flush(IterateDomain const &it_domain) {
            typedef typename kcache_final_flush_indexes< IterationPolicy >::type k_final_flushing_caches_indexes_t;

            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy< IterationPolicy >::value), "error");
            boost::mpl::for_each< k_final_flushing_caches_indexes_t >(
                endpoint_io_cache_functor< IterateDomain, IterationPolicy, cache_io_policy::flush >(
                    it_domain, m_k_caches_tuple));
        }

      private:
        k_caches_tuple_t m_k_caches_tuple;
    };

    template < typename IterateDomainArguments >
    struct is_iterate_domain_cache< iterate_domain_cache< IterateDomainArguments > > : boost::mpl::true_ {};

} // namespace gridtools

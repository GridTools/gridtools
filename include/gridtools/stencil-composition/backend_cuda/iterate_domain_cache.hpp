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

#include <boost/fusion/container/map/convert.hpp>
#include <boost/fusion/include/as_map.hpp>
#include <boost/fusion/include/pair.hpp>
#include <boost/fusion/support/pair.hpp>
#include <boost/mpl/copy_if.hpp>
#include <boost/mpl/count_if.hpp>
#include <boost/mpl/filter_view.hpp>

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/fusion_map_to_mpl_map.hpp"

#include "../accessor_fwd.hpp"
#include "../block.hpp"
#include "../block_size.hpp"
#include "../caches/cache_metafunctions.hpp"
#include "../caches/extract_extent_caches.hpp"
#include "../iterate_domain_fwd.hpp"

#include "./iterate_domain_cache_aux.hpp"

namespace gridtools {

    /**
     * @struct slide_cache_functor
     * functor used to apply the slide operation on all kcache arguments of the kcache tuple
     */
    template <typename IterationPolicy>
    struct slide_cache_functor {
      public:
        GT_FUNCTION
        slide_cache_functor() {}

        // \tparam CacheStoragePair is a pair of <index,cache_storage>
        template <typename CacheStoragePair>
        GT_FUNCTION void operator()(CacheStoragePair &st_pair) const {
            st_pair.second.template slide<IterationPolicy>();
        }
    };

    /**
     * @struct filter_map_indexes
     * metafunction that returns a sequence of all the indexes of the pair elements
     * in the map that fulfil the predicate
     * \tparam Map is a map of <index, cache_storage>
     * \tparam Pred predicate used to filter the map elements
     */
    template <typename Map, template <typename> class Pred>
    struct filter_map_indexes {
        template <typename Pair>
        struct apply_pred {
            typedef typename Pred<typename Pair::second::cache_t>::type type;
        };
        typedef typename boost::mpl::fold<Map,
            boost::mpl::vector0<>,
            boost::mpl::if_<apply_pred<boost::mpl::_2>,
                boost::mpl::push_back<boost::mpl::_1, boost::mpl::first<boost::mpl::_2>>,
                boost::mpl::_1>>::type type;
    };

    /**
     * @class iterate_domain_cache
     * class that provides all the caching functionality needed by the iterate domain.
     * It keeps in type information all the caches setup by the user and provides methods to access cache storage and
     * perform
     * all the caching operations, like filling, sliding or flushing.
     */
    template <typename IterateDomainArguments>
    class iterate_domain_cache {
        DISALLOW_COPY_AND_ASSIGN(iterate_domain_cache);

        GRIDTOOLS_STATIC_ASSERT((is_iterate_domain_arguments<IterateDomainArguments>::value), GT_INTERNAL_ERROR);
        typedef typename IterateDomainArguments::esf_sequence_t esf_sequence_t;
        typedef typename IterateDomainArguments::cache_sequence_t cache_sequence_t;

      private:
        // checks if an arg is used by any of the esfs within a sequence
        template <typename EsfSequence, typename Arg>
        struct is_arg_used_in_esf_sequence {
            typedef typename boost::mpl::fold<EsfSequence,
                boost::mpl::false_,
                boost::mpl::or_<boost::mpl::_1, boost::mpl::contains<esf_args<boost::mpl::_2>, Arg>>>::type type;
        };

        using backend_ids_t = typename IterateDomainArguments::backend_ids_t;
        using block_size_t = block_size<block_i_size(backend_ids_t{}), block_j_size(backend_ids_t{}), 1>;

      public:
        GT_FUNCTION
        iterate_domain_cache() {}

        GT_FUNCTION
        ~iterate_domain_cache() {}

        static constexpr bool has_ij_caches =
            boost::mpl::count_if<cache_sequence_t, cache_is_type<IJ>>::type::value != 0;

        // remove caches which are not used by the stencil stages
        typedef typename boost::mpl::copy_if<cache_sequence_t,
            is_arg_used_in_esf_sequence<esf_sequence_t, cache_parameter<boost::mpl::_>>>::type caches_t;

        // extract a sequence of extents for each ij cache
        typedef typename extract_ij_extents_for_caches<IterateDomainArguments>::type ij_cache_extents_map_t;

        // extract a sequence of extents for each k cache
        typedef typename extract_k_extents_for_caches<IterateDomainArguments>::type k_cache_extents_map_t;
        // compute the fusion vector of pair<index_type, cache_storage> for ij caches
        typedef typename get_cache_storage_tuple<IJ,
            caches_t,
            ij_cache_extents_map_t,
            block_size_t,
            typename IterateDomainArguments::local_domain_t>::type ij_caches_vector_t;

        // compute the fusion vector of pair<index_type, cache_storage> for k caches
        typedef typename get_cache_storage_tuple<K,
            caches_t,
            k_cache_extents_map_t,
            block_size_t,
            typename IterateDomainArguments::local_domain_t>::type k_caches_vector_t;

        // extract a fusion map from the fusion vector of pairs for ij caches
        typedef typename boost::fusion::result_of::as_map<ij_caches_vector_t>::type ij_caches_tuple_t;

        // extract a fusion map from the fusion vector of pairs for k caches
        typedef typename boost::fusion::result_of::as_map<k_caches_vector_t>::type k_caches_tuple_t;

        // compute an mpl from the previous fusion vector, to be used for compile time meta operations
        typedef typename fusion_map_to_mpl_map<ij_caches_tuple_t>::type ij_caches_map_t;

        // compute an mpl from the previous fusion vector, to be used for compile time meta operations
        typedef typename fusion_map_to_mpl_map<k_caches_tuple_t>::type k_caches_map_t;

        // list of indexes of kcaches that require flushing operations
        typedef typename filter_map_indexes<k_caches_map_t, is_flushing_cache>::type k_flushing_caches_indexes_t;

        // list of indexes of kcaches that require end-point flushing operations
        typedef typename filter_map_indexes<k_caches_map_t, is_epflushing_cache>::type k_epflushing_caches_indexes_t;

        // list of indexes of kcaches that require filling operations
        typedef typename filter_map_indexes<k_caches_map_t, is_filling_cache>::type k_filling_caches_indexes_t;

        // list of indexes of kcaches that require begin-point filling operations
        typedef typename filter_map_indexes<k_caches_map_t, is_bpfilling_cache>::type k_bpfilling_caches_indexes_t;

        // set of "bypass" caches
        typedef typename get_cache_set_for_type<bypass, caches_t, typename IterateDomainArguments::local_domain_t>::type
            bypass_caches_set_t;

        // associative container with all caches
        typedef typename get_cache_set<caches_t, typename IterateDomainArguments::local_domain_t>::type all_caches_t;

        // returns a k cache from the tuple
        template <typename IndexType>
        GT_FUNCTION typename boost::mpl::at<k_caches_map_t, IndexType>::type &RESTRICT get_k_cache() {
            GRIDTOOLS_STATIC_ASSERT(
                (boost::mpl::has_key<k_caches_map_t, IndexType>::value), "Accessing a non registered cached");
            return boost::fusion::at_key<IndexType>(m_k_caches_tuple);
        }

        // slide all the k caches
        template <typename IterationPolicy>
        GT_FUNCTION void slide_caches() {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), GT_INTERNAL_ERROR);
            boost::fusion::for_each(m_k_caches_tuple, slide_cache_functor<IterationPolicy>());
        }

        /**
         * fill next k level from main memory for all k caches. The position of the kcache being filled
         * depends on the iteration policy
         * \tparam IterationPolicy forward: backward
         * \param it_domain an iterate domain
         * \param klevel current k level index
         * \param grid a grid with loop bounds information
         */
        template <typename IterationPolicy, typename IterateDomain, typename Grid>
        GT_FUNCTION void fill_caches(IterateDomain const &it_domain, const int_t klevel, const Grid &grid) {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);

            boost::mpl::for_each<k_filling_caches_indexes_t>(_impl::io_cache_functor<k_caches_tuple_t,
                k_caches_map_t,
                IterateDomain,
                IterationPolicy,
                Grid,
                cache_io_policy::fill>(it_domain, m_k_caches_tuple, klevel, grid));
        }

        /**
         * flush the last k level of the ring buffer into main memory. The position of the kcache being flushed
         * depends on the iteration policy
         * \tparam IterationPolicy forward: backward
         * \param it_domain an iterate domain
         * \param klevel current k level index
         * \param grid a grid with loop bounds information
         */
        template <typename IterationPolicy, typename IterateDomain, typename Grid>
        GT_FUNCTION void flush_caches(IterateDomain const &it_domain, const int_t klevel, Grid const &grid) {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);

            boost::mpl::for_each<k_flushing_caches_indexes_t>(_impl::io_cache_functor<k_caches_tuple_t,
                k_caches_map_t,
                IterateDomain,
                IterationPolicy,
                Grid,
                cache_io_policy::flush>(it_domain, m_k_caches_tuple, klevel, grid));
        }

        /**
         * @struct kcache_final_flush_indexes
         * metafunction that computes the list of indexes of all k caches that require a final flush
         */
        template <typename IterationPolicy>
        struct kcache_final_flush_indexes {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), GT_INTERNAL_ERROR);

            /**
             * @brief it determines if a give level is the last level (in a certain iteration order specified by
             * IterationPolicy) of the interval of use of a CacheStorage
             * @tparam CacheStorage cache storage
             */
            template <typename CacheStorage>
            struct is_end_index {
                using cache_t = typename CacheStorage::cache_t;
                using to_index = typename level_to_index<typename IterationPolicy::to>::type;

                static constexpr bool value = (IterationPolicy::value == enumtype::forward)
                                                  ? (interval_to_index<typename cache_t::interval_t>::type::value ==
                                                        level_to_index<typename IterationPolicy::to>::type::value)
                                                  : (interval_from_index<typename cache_t::interval_t>::type::value ==
                                                        level_to_index<typename IterationPolicy::to>::type::value);
            };

            // determine indexes of all k caches that require flushing, whose associated interval ends with the interval
            // of the current iteration
            // policy.
            using interval_flushing_indexes_t = typename boost::mpl::filter_view<k_flushing_caches_indexes_t,
                is_end_index<boost::mpl::at<k_caches_map_t, boost::mpl::_>>>::type;

            // same for those k caches that need an end-point flush. Determine among them, which ones have an interval
            // whose end
            // matches the current interval
            using interval_epflushing_indexes_t = typename boost::mpl::copy_if<k_epflushing_caches_indexes_t,
                is_end_index<boost::mpl::at<k_caches_map_t, boost::mpl::_>>>::type;

            using type = typename boost::mpl::copy<interval_flushing_indexes_t,
                boost::mpl::inserter<interval_epflushing_indexes_t,
                    boost::mpl::push_back<boost::mpl::_1, boost::mpl::_2>>>::type;
        };

        /**
         * @struct kcache_begin_fill_indexes
         * metafunction that computes the list of indexes of all k caches that require a begin pre-fill of the cache
         */
        template <typename IterationPolicy>
        struct kcache_begin_fill_indexes {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), GT_INTERNAL_ERROR);

            /**
             * @brief it determines if a give level is the last level (in a certain iteration order specified by
             * IterationPolicy) of the interval of use of a CacheStorage
             * @tparam CacheStorage cache storage
             */
            template <typename CacheStorage>
            struct is_end_index {
                using cache_t = typename CacheStorage::cache_t;
                using to_index = typename level_to_index<typename IterationPolicy::to>::type;

                static constexpr bool value = (IterationPolicy::value == enumtype::forward)
                                                  ? (interval_from_index<typename cache_t::interval_t>::type::value ==
                                                        level_to_index<typename IterationPolicy::from>::type::value)
                                                  : (interval_to_index<typename cache_t::interval_t>::type::value ==
                                                        level_to_index<typename IterationPolicy::from>::type::value);
            };

            // determine indexes of all k caches that require filling, whose associated interval starts with the
            // interval
            // of the current iteration policy.
            using interval_filling_indexes_t = typename boost::mpl::filter_view<k_filling_caches_indexes_t,
                is_end_index<boost::mpl::at<k_caches_map_t, boost::mpl::_>>>::type;

            // same for those k cache that require a begin-point filling
            using interval_bpfilling_indexes_t = typename boost::mpl::copy_if<k_bpfilling_caches_indexes_t,
                is_end_index<boost::mpl::at<k_caches_map_t, boost::mpl::_>>>::type;

            using type = typename boost::mpl::copy<interval_filling_indexes_t,
                boost::mpl::inserter<interval_bpfilling_indexes_t,
                    boost::mpl::push_back<boost::mpl::_1, boost::mpl::_2>>>::type;
        };

        /**
         * Initial fill of the of the kcaches. Before the iteration over k starts, we need to prefill the k level
         * of the cache with k > 0 (<0) for the forward (backward) iteration policy
         * \tparam IterationPolicy forward: backward
         * \param it_domain an iterate domain
         */
        template <typename IterationPolicy, typename IterateDomain>
        GT_FUNCTION void begin_fill(IterateDomain const &it_domain) {
            typedef typename kcache_begin_fill_indexes<IterationPolicy>::type k_begin_filling_caches_indexes_t;
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), GT_INTERNAL_ERROR);

            boost::mpl::for_each<k_begin_filling_caches_indexes_t>(_impl::endpoint_io_cache_functor<k_caches_tuple_t,
                k_caches_map_t,
                IterateDomain,
                IterationPolicy,
                cache_io_policy::fill>(it_domain, m_k_caches_tuple));
        }

        /**
         * Final flush of the of the kcaches. After the iteration over k is done, we still need to flush the remaining
         * k levels of the cache with k > 0 (<0) for the backward (forward) iteration policy
         * \tparam IterationPolicy forward: backward
         * \param it_domain an iterate domain
         */
        template <typename IterationPolicy, typename IterateDomain>
        GT_FUNCTION void final_flush(IterateDomain const &it_domain) {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), GT_INTERNAL_ERROR);

            typedef typename kcache_final_flush_indexes<IterationPolicy>::type k_final_flushing_caches_indexes_t;

            boost::mpl::for_each<k_final_flushing_caches_indexes_t>(_impl::endpoint_io_cache_functor<k_caches_tuple_t,
                k_caches_map_t,
                IterateDomain,
                IterationPolicy,
                cache_io_policy::flush>(it_domain, m_k_caches_tuple));
        }

      private:
        k_caches_tuple_t m_k_caches_tuple;
    };

} // namespace gridtools

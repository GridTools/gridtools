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
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/transform_view.hpp>

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/fusion_map_to_mpl_map.hpp"

#include "../block.hpp"
#include "../block_size.hpp"
#include "../caches/cache_metafunctions.hpp"
#include "../caches/extract_extent_caches.hpp"
#include "../iterate_domain_fwd.hpp"

#include "./iterate_domain_cache_aux.hpp"

namespace gridtools {
    namespace impl_ {

        /**
         * @brief Boost MPL-style metafunction class for filtering cache index sequences.
         * @tparam CacheMap Boost MPL map of indices to cache storages.
         * @tparam Pred Boost MPL lambda expression or metafunction class. Filer predicate, taking a single
         * cache_storage as input.
         * @tparam Indexes Sequence of indexes to consider, by default all indexes in CacheMap.
         */
        template <class CacheMap,
            class Pred,
            class Indexes = typename boost::mpl::transform_view<CacheMap, boost::mpl::first<boost::mpl::_>>::type>
        struct filter_indexes {
            using type = typename boost::mpl::filter_view<Indexes,
                typename Pred::template apply<boost::mpl::at<CacheMap, boost::mpl::_>>>::type;
        };

        /**
         * @brief Metafunction returning all indices in `CacheMap` whose associated cache matches the predicate `Pred`.
         * @tparam CacheMap Boost MPL map of indices to cache storages.
         * @tparam Pred Prediacate meta function, taking a single cache (actually struct gridtools::detail::cache_impl)
         * as input.
         */
        template <class CacheMap, template <class> class Pred>
        struct get_indexes_by_cache {
          private:
            template <class CacheStorage>
            struct pred : Pred<typename CacheStorage::cache_t> {};

          public:
            using type = typename filter_indexes<CacheMap, boost::mpl::quote1<pred>>::type;
        };

        /**
         * @brief Boost MPL-style metafunction class, evaluating if `CacheStorage` interval end point matches with
         * `IterationPolicy` end point.
         */
        template <class IterationPolicy>
        struct is_end_index {
            static constexpr auto iteration_to_index = level_to_index<typename IterationPolicy::to>::type::value;

            template <class CacheStorage>
            struct apply {
                using cache_interval_t = typename CacheStorage::cache_t::interval_t;
                static constexpr auto from_index = interval_from_index<cache_interval_t>::type::value;
                static constexpr auto to_index = interval_to_index<cache_interval_t>::type::value;
                static constexpr bool value = (IterationPolicy::value == enumtype::forward)
                                                  ? to_index == iteration_to_index
                                                  : from_index == iteration_to_index;
            };
        };

        /**
         * @brief Boost MPL-style metafunction class, evaluating if `CacheStorage` interval begin point matches with
         * `IterationPolicy` begin point.
         */
        template <class IterationPolicy>
        struct is_begin_index {
            static constexpr auto iteration_from_index = level_to_index<typename IterationPolicy::from>::type::value;

            template <class CacheStorage>
            struct apply {
                using cache_interval_t = typename CacheStorage::cache_t::interval_t;
                static constexpr auto from_index = interval_from_index<cache_interval_t>::type::value;
                static constexpr auto to_index = interval_to_index<cache_interval_t>::type::value;
                static constexpr bool value = (IterationPolicy::value == enumtype::forward)
                                                  ? from_index == iteration_from_index
                                                  : to_index == iteration_from_index;
            };
        };

        /**
         * @brief Boost MPL-style metafunction class, evaluating if `CacheStorage` interval overlaps with
         * `IterationPolicy` interval.
         */
        template <class IterationPolicy>
        struct is_active_index {
            static constexpr auto iteration_from_index = level_to_index<typename IterationPolicy::from>::type::value;
            static constexpr auto iteration_to_index = level_to_index<typename IterationPolicy::to>::type::value;

            template <class CacheStorage>
            struct apply {
                using cache_interval_t = typename CacheStorage::cache_t::interval_t;
                static constexpr auto from_index = interval_from_index<cache_interval_t>::type::value;
                static constexpr auto to_index = interval_to_index<cache_interval_t>::type::value;
                static constexpr bool value = (to_index >= iteration_from_index) && (from_index <= iteration_to_index);
            };
        };

        /**
         * @brief Functor used to apply the slide operation on all kcache arguments of the kcache tuple.
         */
        template <typename IterationPolicy>
        struct slide_cache_functor {
            /*
             * @tparam CacheStoragePair is a pair of <index, cache_storage>
             */
            template <typename CacheStoragePair>
            GT_FUNCTION void operator()(CacheStoragePair &st_pair) const {
                st_pair.second.template slide<IterationPolicy>();
            }
        };
    } // namespace impl_

    /**
     * @class iterate_domain_cache
     * class that provides all the caching functionality needed by the iterate domain.
     * It keeps in type information all the caches setup by the user and provides methods to access cache storage and
     * perform
     * all the caching operations, like filling, sliding or flushing.
     */
    template <typename IterateDomainArguments>
    class iterate_domain_cache {
        GT_DISALLOW_COPY_AND_ASSIGN(iterate_domain_cache);

        GT_STATIC_ASSERT((is_iterate_domain_arguments<IterateDomainArguments>::value), GT_INTERNAL_ERROR);
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
            boost::mpl::count_if<cache_sequence_t, cache_is_type<cache_type::IJ>>::type::value != 0;

        // remove caches which are not used by the stencil stages
        typedef typename boost::mpl::copy_if<cache_sequence_t,
            is_arg_used_in_esf_sequence<esf_sequence_t, cache_parameter<boost::mpl::_>>>::type caches_t;

        // extract a sequence of extents for each ij cache
        typedef typename extract_ij_extents_for_caches<IterateDomainArguments>::type ij_cache_extents_map_t;

        // extract a sequence of extents for each k cache
        typedef typename extract_k_extents_for_caches<IterateDomainArguments>::type k_cache_extents_map_t;
        // compute the fusion vector of pair<index_type, cache_storage> for ij caches
        typedef typename get_cache_storage_tuple<cache_type::IJ,
            caches_t,
            ij_cache_extents_map_t,
            block_size_t,
            typename IterateDomainArguments::local_domain_t>::type ij_caches_vector_t;

        // compute the fusion vector of pair<index_type, cache_storage> for k caches
        typedef typename get_cache_storage_tuple<cache_type::K,
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
        typedef
            typename impl_::get_indexes_by_cache<k_caches_map_t, is_flushing_cache>::type k_flushing_caches_indexes_t;

        // list of indexes of kcaches that require filling operations
        typedef typename impl_::get_indexes_by_cache<k_caches_map_t, is_filling_cache>::type k_filling_caches_indexes_t;

        // associative container with all caches
        typedef typename get_cache_set<caches_t, typename IterateDomainArguments::local_domain_t>::type all_caches_t;

        // returns a k cache from the tuple
        template <typename IndexType>
        GT_FUNCTION typename boost::mpl::at<k_caches_map_t, IndexType>::type &RESTRICT get_k_cache() const {
            GT_STATIC_ASSERT(
                (boost::mpl::has_key<k_caches_map_t, IndexType>::value), "Accessing a non registered cached");
            return boost::fusion::at_key<IndexType>(const_cast<k_caches_tuple_t &>(m_k_caches_tuple));
        }

        // slide all the k caches
        template <typename IterationPolicy>
        GT_FUNCTION void slide_caches() {
            GT_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), GT_INTERNAL_ERROR);
            boost::fusion::for_each(m_k_caches_tuple, impl_::slide_cache_functor<IterationPolicy>());
        }

        /**
         * fill next k level from main memory for all k caches. The position of the kcache being filled
         * depends on the iteration policy
         * \tparam IterationPolicy forward: backward
         * \param it_domain an iterate domain
         * \param first_level indicates that this function is called the first time in the k-loop
         */
        template <typename IterationPolicy, typename IterateDomain>
        GT_FUNCTION void fill_caches(IterateDomain const &it_domain, bool first_level) {
            GT_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), GT_INTERNAL_ERROR);

            if (first_level) {
                boost::mpl::for_each<k_filling_caches_indexes_t>(_impl::endpoint_io_cache_functor<k_caches_tuple_t,
                    k_caches_map_t,
                    IterateDomain,
                    IterationPolicy,
                    cache_io_policy::fill>(it_domain, m_k_caches_tuple));
            }

            boost::mpl::for_each<k_filling_caches_indexes_t>(_impl::io_cache_functor<k_caches_tuple_t,
                k_caches_map_t,
                IterateDomain,
                IterationPolicy,
                cache_io_policy::fill>(it_domain, m_k_caches_tuple));
        }

        /**
         * flush the last k level of the ring buffer into main memory. The position of the kcache being flushed
         * depends on the iteration policy
         * \tparam IterationPolicy forward: backward
         * \param it_domain an iterate domain
         * \param last_level indicates that this function is called the last time in the k-loop
         */
        template <typename IterationPolicy, typename IterateDomain>
        GT_FUNCTION void flush_caches(IterateDomain const &it_domain, bool last_level) {
            GT_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), GT_INTERNAL_ERROR);

            boost::mpl::for_each<k_flushing_caches_indexes_t>(_impl::io_cache_functor<k_caches_tuple_t,
                k_caches_map_t,
                IterateDomain,
                IterationPolicy,
                cache_io_policy::flush>(it_domain, m_k_caches_tuple));

            if (last_level) {
                boost::mpl::for_each<k_flushing_caches_indexes_t>(_impl::endpoint_io_cache_functor<k_caches_tuple_t,
                    k_caches_map_t,
                    IterateDomain,
                    IterationPolicy,
                    cache_io_policy::flush>(it_domain, m_k_caches_tuple));
            }
        }

      private:
        k_caches_tuple_t m_k_caches_tuple;
    };

} // namespace gridtools

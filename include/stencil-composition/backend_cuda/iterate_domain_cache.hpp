/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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

#include "common/defs.hpp"
#include <boost/fusion/container/map/convert.hpp>
#include <boost/fusion/include/as_map.hpp>
#include <boost/fusion/support/pair.hpp>
#include <boost/fusion/include/pair.hpp>
#include <boost/mpl/copy_if.hpp>
#include <boost/mpl/filter_view.hpp>
#include "common/generic_metafunctions/vector_to_map.hpp"
#include "common/generic_metafunctions/fusion_map_to_mpl_map.hpp"
#include "stencil-composition/iterate_domain_fwd.hpp"
#include "../caches/cache_metafunctions.hpp"
#include "../caches/extract_extent_caches.hpp"
#include "../accessor_fwd.hpp"
#include "../../common/generic_metafunctions/vector_to_vector.hpp"

namespace gridtools {

    template < typename IterationPolicy >
    struct slide_cache_functor {
        // TODO KCACHE use lambda
      public:
        slide_cache_functor() {}

        /** @brief operator inserting a storage raw pointer

            filters out the arguments which are not of storage type (and thus do not have an associated metadata)
         */
        template < typename Arg >
        void operator()(Arg &arg_) const {
            arg_.second.template slide< IterationPolicy >();
        }
    };

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

    template < typename AccIndex, enumtype::execution ExecutionPolicy >
    struct sync_mem_accessor {
        template < int_t Offset >
        struct apply_t {
            template < typename IterateDomain, typename CacheStorage >
            GT_FUNCTION static int_t apply(IterateDomain const &it_domain, CacheStorage const &cache_st) {

                typedef accessor< AccIndex::value, enumtype::inout, extent< 0, 0, 0, 0, -Offset - 1, Offset + 1 > >
                    acc_t;
                constexpr acc_t acc_(0, 0, (ExecutionPolicy == enumtype::forward) ? -Offset - 1 : Offset + 1);

                it_domain.gmem_access(acc_) = cache_st.at(acc_);
                return 0;
            }
        };
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

        GRIDTOOLS_STATIC_ASSERT(
            (is_iterate_domain_arguments< IterateDomainArguments >::value), "Internal error: wrong type");
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

        // extract a sequence of extents for each cache
        typedef typename extract_ij_extents_for_caches< IterateDomainArguments >::type ij_cache_extents_map_t;

        // extract a sequence of extents for each cache
        typedef typename extract_k_extents_for_caches< IterateDomainArguments >::type k_cache_extents_map_t;

        // compute the fusion vector of pair<index_type, cache_storage>
        typedef typename get_cache_storage_tuple< IJ,
            caches_t,
            ij_cache_extents_map_t,
            typename IterateDomainArguments::physical_domain_block_size_t,
            typename IterateDomainArguments::local_domain_t >::type ij_caches_vector_t;

        // compute the fusion vector of pair<index_type, cache_storage>
        typedef typename get_cache_storage_tuple< K,
            caches_t,
            k_cache_extents_map_t,
            typename IterateDomainArguments::physical_domain_block_size_t,
            typename IterateDomainArguments::local_domain_t >::type k_caches_vector_t;

        // extract a fusion map from the fusion vector of pairs
        typedef typename boost::fusion::result_of::as_map< ij_caches_vector_t >::type ij_caches_tuple_t;

        // extract a fusion map from the fusion vector of pairs
        typedef typename boost::fusion::result_of::as_map< k_caches_vector_t >::type k_caches_tuple_t;

        // compute an mpl from the previous fusion vector, to be used for compile time meta operations
        typedef typename fusion_map_to_mpl_map< ij_caches_tuple_t >::type ij_caches_map_t;

        // compute an mpl from the previous fusion vector, to be used for compile time meta operations
        typedef typename fusion_map_to_mpl_map< k_caches_tuple_t >::type k_caches_map_t;

        typedef typename filter_map_indexes< k_caches_map_t, is_flushing_cache >::type k_flushing_caches_indexes_t;

        typedef typename filter_map_indexes< k_caches_map_t, is_epflushing_cache >::type k_epflushing_caches_indexes_t;

        typedef
            typename get_cache_set_for_type< bypass, caches_t, typename IterateDomainArguments::local_domain_t >::type
                bypass_caches_set_t;

        // associative container with all caches
        typedef typename get_cache_set< caches_t, typename IterateDomainArguments::local_domain_t >::type all_caches_t;

        template < typename IndexType >
        GT_FUNCTION typename boost::mpl::at< k_caches_map_t, IndexType >::type &RESTRICT get_k_cache() {
            GRIDTOOLS_STATIC_ASSERT(
                (boost::mpl::has_key< k_caches_map_t, IndexType >::value), "Accessing a non registered cached");
            return boost::fusion::at_key< IndexType >(m_k_caches_tuple);
        }

        template < typename IterationPolicy >
        GT_FUNCTION void slide_caches() {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy< IterationPolicy >::value), "error");

            // copy of the non-tmp metadata into m_metadata_set
            boost::fusion::for_each(m_k_caches_tuple, slide_cache_functor< IterationPolicy >());
        }

        template < typename IterateDomain, typename IterationPolicy >
        struct flushing_functor {

            GT_FUNCTION
            flushing_functor(IterateDomain const &it_domain, k_caches_tuple_t const &kcaches)
                : m_it_domain(it_domain), m_kcaches(kcaches) {}

            IterateDomain const &m_it_domain;
            k_caches_tuple_t const &m_kcaches;

            template < typename Idx >
            GT_FUNCTION void operator()(Idx const &) const {
#ifdef CXX11_ENABLED
                typedef typename boost::mpl::at< k_caches_map_t, Idx >::type k_cache_storage_t;

                constexpr int_t kminus =
                    (IterationPolicy::value == enumtype::forward)
                        ? boost::mpl::at_c< typename k_cache_storage_t::minus_t::type, 2 >::type::value
                        : 0;

                constexpr int_t kplus =
                    (IterationPolicy::value == enumtype::backward)
                        ? boost::mpl::at_c< typename k_cache_storage_t::plus_t::type, 2 >::type::value
                        : 0;

                constexpr int_t koffset = (IterationPolicy::value == enumtype::forward) ? kminus : kplus;

                typedef accessor< Idx::value, enumtype::inout, extent< 0, 0, 0, 0, kminus, kplus > > acc_t;
                constexpr acc_t acc_(0, 0, koffset);

                m_it_domain.gmem_access(acc_) = boost::fusion::at_key< Idx >(m_kcaches).at(acc_);
#endif
            }
        };

        template < typename IterateDomain, typename IterationPolicy >
        struct final_flushing_functor {

            GT_FUNCTION
            final_flushing_functor(IterateDomain const &it_domain, k_caches_tuple_t const &kcaches)
                : m_it_domain(it_domain), m_kcaches(kcaches) {}

            IterateDomain const &m_it_domain;
            k_caches_tuple_t const &m_kcaches;

            template < typename Idx >
            GT_FUNCTION void operator()(Idx const &) const {
#ifdef CXX11_ENABLED
                typedef typename boost::mpl::at< k_caches_map_t, Idx >::type k_cache_storage_t;

                GRIDTOOLS_STATIC_ASSERT(
                    (boost::mpl::at_c< typename k_cache_storage_t::minus_t::type, 2 >::type::value <= 0 &&
                        boost::mpl::at_c< typename k_cache_storage_t::plus_t::type, 2 >::type::value >= 0),
                    "Error");

                constexpr uint_t koffset =
                    (IterationPolicy::value == enumtype::forward)
                        ? -boost::mpl::at_c< typename k_cache_storage_t::minus_t::type, 2 >::type::value
                        : boost::mpl::at_c< typename k_cache_storage_t::plus_t::type, 2 >::type::value;

                using seq = gridtools::apply_gt_integer_sequence<
                    typename gridtools::make_gt_integer_sequence< int_t, koffset >::type >;

                seq::template apply_void_lambda< sync_mem_accessor< Idx, IterationPolicy::value >::apply_t >(
                    m_it_domain, boost::fusion::at_key< Idx >(m_kcaches));
#endif
            }
        };

        template < typename IterationPolicy, typename IterateDomain >
        GT_FUNCTION void flush_caches(IterateDomain const &it_domain) {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy< IterationPolicy >::value), "error");
            // copy of the non-tmp metadata into m_metadafromfromfromfromta_set
            boost::mpl::for_each< k_flushing_caches_indexes_t >(
                flushing_functor< IterateDomain, IterationPolicy >(it_domain, m_k_caches_tuple));
        }

        template < typename IterationPolicy >
        struct kcache_final_flush_indexes {
#ifdef CXX11_ENABLED
            template < typename CacheStorage >
            struct is_end_index {
                //                GRIDTOOLS_STATIC_ASSERT((is_cache< Cache >::value), "Internal Error");
                using cache_t = typename CacheStorage::cache_t;
                using to_index = typename level_to_index< typename IterationPolicy::to >::type;

                static constexpr bool value = (IterationPolicy::value == enumtype::forward)
                                                  ? (interval_to_index< typename cache_t::interval_t >::type::value ==
                                                        level_to_index< typename IterationPolicy::to >::type::value)
                                                  : (interval_from_index< typename cache_t::interval_t >::type::value ==
                                                        level_to_index< typename IterationPolicy::to >::type::value);
            };

            using interval_flushing_indexes_t = typename boost::mpl::filter_view< k_flushing_caches_indexes_t,
                is_end_index< boost::mpl::at< k_caches_map_t, boost::mpl::_ > > >::type;

            using interval_epflushing_indexes_t =
                typename vector_to_vector< typename boost::mpl::filter_view< k_epflushing_caches_indexes_t,
                    is_end_index< boost::mpl::at< k_caches_map_t, boost::mpl::_ > > >::type >::type;

            using type =
                typename boost::mpl::copy< interval_flushing_indexes_t,
                    boost::mpl::inserter< interval_epflushing_indexes_t,
                                               boost::mpl::push_back< boost::mpl::_1, boost::mpl::_2 > > >::type;

#endif
        };

        template < typename IterationPolicy, typename IterateDomain >
        GT_FUNCTION void final_flush(IterateDomain const &it_domain) {
            typedef typename kcache_final_flush_indexes< IterationPolicy >::type k_final_flushing_caches_indexes_t;

            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy< IterationPolicy >::value), "error");
            boost::mpl::for_each< k_final_flushing_caches_indexes_t >(
                final_flushing_functor< IterateDomain, IterationPolicy >(it_domain, m_k_caches_tuple));
        }

      private:
        k_caches_tuple_t m_k_caches_tuple;
    };

    template < typename IterateDomainArguments >
    struct is_iterate_domain_cache< iterate_domain_cache< IterateDomainArguments > > : boost::mpl::true_ {};

} // namespace gridtools

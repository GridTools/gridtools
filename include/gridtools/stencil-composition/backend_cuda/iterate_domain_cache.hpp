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

#include <boost/fusion/include/as_map.hpp>
#include <boost/fusion/include/at_key.hpp>

#include "../../common/defs.hpp"
#include "../../meta.hpp"
#include "../block.hpp"
#include "../caches/cache_metafunctions.hpp"
#include "../caches/extract_extent_caches.hpp"
#include "../iteration_policy.hpp"
#include "../run_functor_arguments.hpp"
#include "./iterate_domain_cache_aux.hpp"

namespace gridtools {
    /**
     * @class iterate_domain_cache
     * class that provides all the caching functionality needed by the iterate domain.
     * It keeps in type information all the caches setup by the user and provides methods to access cache storage and
     * perform all the caching operations, like filling, sliding or flushing.
     */
    template <typename IterateDomainArguments>
    class iterate_domain_cache {
        GT_STATIC_ASSERT(is_iterate_domain_arguments<IterateDomainArguments>::value, GT_INTERNAL_ERROR);

        using cache_sequence_t = typename IterateDomainArguments::cache_sequence_t;

        using backend_ids_t = typename IterateDomainArguments::backend_ids_t;

        // compute the fusion vector of pair<index_type, cache_storage> for ij caches
        typedef typename get_ij_cache_storage_tuple<cache_sequence_t,
            typename IterateDomainArguments::max_extent_for_tmp_t,
            block_i_size(backend_ids_t{}),
            block_j_size(backend_ids_t{})>::type ij_caches_vector_t;

        using k_caches_tuple_t =
            typename boost::fusion::result_of::as_map<typename get_k_cache_storage_tuple<cache_sequence_t,
                typename IterateDomainArguments::esf_sequence_t>::type>::type;

        k_caches_tuple_t m_k_caches_tuple;

      public:
        // extract a fusion map from the fusion vector of pairs for ij caches
        using ij_caches_tuple_t = typename boost::fusion::result_of::as_map<ij_caches_vector_t>::type;

        template <class Arg, class Accessor>
        GT_FUNCTION auto get_k_cache(Accessor const &acc) const
            GT_AUTO_RETURN(boost::fusion::at_key<Arg>(const_cast<k_caches_tuple_t &>(m_k_caches_tuple)).at(acc));

        // slide all the k caches
        template <class IterationPolicy>
        GT_FUNCTION void slide_caches() {
            GT_STATIC_ASSERT(is_iteration_policy<IterationPolicy>::value, GT_INTERNAL_ERROR);

            using k_caches_t = GT_META_CALL(
                meta::transform, (cache_parameter, GT_META_CALL(meta::filter, (is_k_cache, cache_sequence_t))));

            _impl::slide_caches<k_caches_t, IterationPolicy::value>(m_k_caches_tuple);
        }

        /**
         * fill next k level from main memory for all k caches. The position of the kcache being filled
         * depends on the iteration policy
         * \tparam IterationPolicy forward: backward
         * \param it_domain an iterate domain
         * \param first_level indicates that this function is called the first time in the k-loop
         */
        template <class IterationPolicy, class IterateDomain>
        GT_FUNCTION void fill_caches(IterateDomain const &it_domain, bool first_level, array<int_t, 2> validity) {
            GT_STATIC_ASSERT(is_iteration_policy<IterationPolicy>::value, GT_INTERNAL_ERROR);

            using filling_cache_args_t = GT_META_CALL(
                meta::transform, (cache_parameter, GT_META_CALL(meta::filter, (is_filling_cache, cache_sequence_t))));

            _impl::sync_caches<filling_cache_args_t, IterationPolicy::value, sync_type::fill>(
                it_domain, m_k_caches_tuple, first_level, validity);
        }

        /**
         * flush the last k level of the ring buffer into main memory. The position of the kcache being flushed
         * depends on the iteration policy
         * \tparam IterationPolicy forward: backward
         * \param it_domain an iterate domain
         * \param last_level indicates that this function is called the last time in the k-loop
         */
        template <typename IterationPolicy, typename IterateDomain>
        GT_FUNCTION void flush_caches(IterateDomain const &it_domain, bool last_level, array<int_t, 2> validity) {
            GT_STATIC_ASSERT(is_iteration_policy<IterationPolicy>::value, GT_INTERNAL_ERROR);

            using flushing_cache_args_t = GT_META_CALL(
                meta::transform, (cache_parameter, GT_META_CALL(meta::filter, (is_flushing_cache, cache_sequence_t))));

            _impl::sync_caches<flushing_cache_args_t, IterationPolicy::value, sync_type::flush>(
                it_domain, m_k_caches_tuple, last_level, validity);
        }
    };
} // namespace gridtools

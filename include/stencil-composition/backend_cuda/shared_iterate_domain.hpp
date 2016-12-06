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
/**@file
   @brief file with classes to store the data members of the iterate domain
   that will be allocated in shared memory
 */
#pragma once
#include <boost/mpl/has_key.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/at.hpp>
#include <boost/fusion/sequence/intrinsic/at_key.hpp>
#include <boost/fusion/include/at_key.hpp>
#include "stencil-composition/accessor.hpp"
#include "common/generic_metafunctions/fusion_map_to_mpl_map.hpp"

namespace gridtools {

    /**
     * @class shared_iterate_domain
     * data structure that holds data members of the iterate domain that must be stored in shared memory.
     * @tparam DataPointerArray array of data pointers
     * @tparam StridesType strides cached type
     * @tparam IJCachesTuple fusion map of <index_type, cache_storage>
     */
    template < typename IJCachesTuple >
    class shared_iterate_domain {
        DISALLOW_COPY_AND_ASSIGN(shared_iterate_domain);
        // TODO: protect IJCachesTuple

      private:
        IJCachesTuple m_ij_caches_tuple;

        // For some reasons fusion metafunctions (such as result_of::at_key) fail on a fusion map
        // constructed with the result_of::as_map from a fusion vector.
        // Therefore we construct here a mirror metadata mpl map type to be used for meta algorithms
        typedef typename fusion_map_to_mpl_map< IJCachesTuple >::type ij_caches_map_t;

      public:
        shared_iterate_domain() {}

        template < typename IndexType >
        GT_FUNCTION typename boost::mpl::at< ij_caches_map_t, IndexType >::type &RESTRICT get_ij_cache() {
            GRIDTOOLS_STATIC_ASSERT(
                (boost::mpl::has_key< ij_caches_map_t, IndexType >::value), "Accessing a non registered cached");
            return boost::fusion::at_key< IndexType >(m_ij_caches_tuple);
        }
    };

} // namespace gridtools

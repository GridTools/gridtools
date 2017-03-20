/*
  GridTools Libraries

  Copyright (c) 2017, GridTools Consortium
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
#pragma once
#include "common/generic_metafunctions/mpl_vector_flatten.hpp"
#include "common/generic_metafunctions/variadic_to_vector.hpp"
#include "stencil-composition/caches/cache.hpp"

namespace gridtools {

    /**
     * function that captures the list of caches provided by the user for a stencil
     */
    template < typename... CacheSequences >
    typename flatten< typename variadic_to_vector< CacheSequences... >::type >::type define_caches(
        CacheSequences &&... caches) {
        // the call to define_caches might gets a variadic list of cache sequences as input
        // (e.g., define_caches(cache<IJ, local>(p_flx(), p_fly()), cache<K, fill>(p_in())); ).
        // Therefore we have to merge the cache sequences into one single mpl vector.
        typedef typename flatten< typename variadic_to_vector< CacheSequences... >::type >::type cache_sequence_t;
        // perform a check if all elements in the merged vector are cache types
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< cache_sequence_t, is_cache >::value),
            "Error: did not provide a sequence of caches to define_caches syntax");

        return cache_sequence_t();
    }

} // namespace gridtools

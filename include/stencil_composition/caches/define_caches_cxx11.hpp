#pragma once
#include "common/generic_metafunctions/mpl_vector_flatten.hpp"
#include "common/generic_metafunctions/variadic_to_vector.hpp"
#include "stencil_composition/caches/cache.hpp"

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

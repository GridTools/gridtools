#pragma once
#include "common/generic_metafunctions/variadic_to_vector.hpp"
#include "stencil-composition/caches/cache.hpp"

namespace gridtools {

    /**
     * function that captures the list of caches provided by the user for a stencil
     */
    template <typename ... CacheTypes>
    typename variadic_to_vector<CacheTypes ... >::type
    define_caches(CacheTypes&& ... caches){
        //although a mpl vector can be directly constructed from variadic templates, we go here through
        // a metafunction, because this constructor of an mpl vector fails when one of the variadic
        // templates is a mpl vector
        typedef typename variadic_to_vector<CacheTypes ... >::type cache_sequence_t;
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<cache_sequence_t, is_cache>::value),
                        "Error: did not provide a sequence of caches to define_caches syntax");
        return cache_sequence_t();
    }

} // namespace gridtools

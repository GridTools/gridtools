#pragma once

#include "cache.hpp"

namespace gridtools {

    /**
     * function that captures the list of caches provided by the user for a stencil
     */
    template <typename ... CacheTypes>
    boost::mpl::vector<CacheTypes>
    define_caches(CacheTypes&& ... caches){
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<boost::mpl::vector<CacheTypes>, is_cache>::value));
        return boost::mpl::vector<CacheTypes ...> >();
    }

} // namespace gridtools

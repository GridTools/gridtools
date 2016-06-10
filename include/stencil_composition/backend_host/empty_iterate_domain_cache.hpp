#pragma once
#include <boost/mpl/map.hpp>

namespace gridtools {

    /**
     * @brief The empty_iterate_domain_cache struct
     * empty implementation of an iterate domain cache, as host backend does not use caches
     */
    struct empty_iterate_domain_cache {
        typedef boost::mpl::map0<> ij_caches_map_t;
        typedef boost::mpl::set0<> all_caches_t;
        typedef boost::mpl::set0<> bypass_caches_set_t;
    };

    template <>
    struct is_iterate_domain_cache< empty_iterate_domain_cache > : boost::mpl::true_ {};
}

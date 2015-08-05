#pragma once
#include <boost/mpl/map.hpp>

namespace gridtools
{

/**
 * @brief The empty_iterate_domain_cache struct
 * empty implementation of an iterate domain cache, as host backend does not uses caches
 */
struct empty_iterate_domain_cache{
    typedef boost::mpl::map0<> ij_caches_map_t;
};

}

/*
 * empty_iterate_domain_cache.hpp
 *
 *  Created on: Jul 21, 2015
 *      Author: cosuna
 */

#pragma once
#include <boost/mpl/map.hpp>

namespace gridtools
{

struct empty_iterate_domain_cache{
    typedef boost::mpl::map0<> ij_caches_map_t;
};

}

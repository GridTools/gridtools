/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
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

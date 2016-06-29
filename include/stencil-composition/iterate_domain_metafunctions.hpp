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
#include "iterate_domain.hpp"

namespace gridtools {
    template < typename T >
    struct positional_iterate_domain;

    template < typename T >
    struct is_iterate_domain : boost::mpl::false_ {};

    template < typename IterateDomainArguments >
    struct is_iterate_domain< iterate_domain< IterateDomainArguments > > : boost::mpl::true_ {};

    template < typename IterateDomainArguments >
    struct is_iterate_domain< positional_iterate_domain< IterateDomainArguments > > : boost::mpl::true_ {};

    template < typename T >
    struct is_positional_iterate_domain : boost::mpl::false_ {};

    template < typename IterateDomainArguments >
    struct is_positional_iterate_domain< positional_iterate_domain< IterateDomainArguments > > : boost::mpl::true_ {};

    template < typename T >
    struct iterate_domain_local_domain;

    template < template < template < class > class, typename > class IterateDomainImpl,
        template < class > class IterateDomainBase,
        typename IterateDomainArguments >
    struct iterate_domain_local_domain< IterateDomainImpl< IterateDomainBase, IterateDomainArguments > > {
        GRIDTOOLS_STATIC_ASSERT(
            (is_iterate_domain< IterateDomainImpl< IterateDomainBase, IterateDomainArguments > >::value),
            "Internal Error: wrong type");
        typedef typename IterateDomainArguments::local_domain_t type;
    };
}

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
#include "run_functor_arguments_fwd.hpp"

namespace gridtools {
    template < typename T >
    struct iterate_domain_local_domain;

    template < typename T >
    struct is_iterate_domain;

    template < typename T >
    struct iterate_domain_impl_ij_caches_map;

    template < typename Impl >
    struct iterate_domain_impl_arguments;

    template < typename IterateDomainArguments,
        template < typename > class IterateDomainBase,
        template < template < typename > class, typename > class IterateDomainImpl >
    struct iterate_domain_impl_arguments< IterateDomainImpl< IterateDomainBase, IterateDomainArguments > > {
        GRIDTOOLS_STATIC_ASSERT(
            (is_iterate_domain_arguments< IterateDomainArguments >::value), "Internal Error: wrong type");
        typedef IterateDomainArguments type;
    };
}

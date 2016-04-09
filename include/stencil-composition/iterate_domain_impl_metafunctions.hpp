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

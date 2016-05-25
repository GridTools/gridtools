#pragma once

namespace gridtools {
    template < typename T >
    struct iterate_domain_local_domain;

    template < typename T >
    struct is_iterate_domain;

    template < typename T >
    struct is_positional_iterate_domain;

    template < typename T >
    struct is_iterate_domain_cache : boost::mpl::false_ {};

    template < typename IterateDomainImpl >
    struct iterate_domain_backend_id;

} // namespace gridtools

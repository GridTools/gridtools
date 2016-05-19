#pragma once
#include "iterate_domain_remapper.hpp"
#include "iterate_domain.hpp"

namespace gridtools {
    namespace icgrid {
        template < typename IterateDomain >
        struct get_grid_topology
        {
            GRIDTOOLS_STATIC_ASSERT((is_iterate_domain<IterateDomain>::value), "Error: type passed is not an iterate domain");
            typedef typename IterateDomain::grid_topology_t type;
        };

        template < typename IterateDomain, typename EsfArgsMap >
        struct get_grid_topology<iterate_domain_remapper<IterateDomain, EsfArgsMap> >
        {
            typedef typename IterateDomain::grid_topology_t type;
        };

        template < typename IterateDomainImpl >
        struct get_grid_topology< iterate_domain< IterateDomainImpl > > {
            typedef typename iterate_domain< IterateDomainImpl >::grid_topology_t type;
        };
    }

} // namespace gridtools

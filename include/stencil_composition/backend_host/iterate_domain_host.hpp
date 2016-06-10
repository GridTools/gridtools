#pragma once

#ifdef STRUCTURED_GRIDS
#include "stencil_composition/structured_grids/backend_host/iterate_domain_host.hpp"
#else
#include "stencil_composition/icosahedral_grids/backend_host/iterate_domain_host.hpp"
#endif

#include "../iterate_domain_fwd.hpp"

namespace gridtools {
    template < template < class > class IterateDomainBase, typename IterateDomainArguments >
    struct iterate_domain_backend_id< iterate_domain_host< IterateDomainBase, IterateDomainArguments > > {
        typedef enumtype::enum_type< enumtype::platform, enumtype::Host > type;
    };
}

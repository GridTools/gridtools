#pragma once

#ifdef STRUCTURED_GRIDS
#include "../structured_grids/backend_cuda/iterate_domain_cuda.hpp"
#else
#include "../icosahedral_grids/backend_cuda/iterate_domain_cuda.hpp"
#endif

#include "../iterate_domain_fwd.hpp"

namespace gridtools {
    template < template < class > class IterateDomainBase, typename IterateDomainArguments >
    struct iterate_domain_backend_id< iterate_domain_cuda< IterateDomainBase, IterateDomainArguments > > {
        typedef enumtype::enum_type< enumtype::platform, enumtype::Cuda > type;
    };
}

#pragma once

#include "intermediate.h"

namespace gridtools {
    template <typename t_backend, typename t_mss_type, typename t_domain, typename t_coords>
    computation* make_computation(t_mss_type const& mss, t_domain & domain, t_coords const& coords) {
        return new intermediate<t_backend, t_mss_type, t_domain, t_coords>(mss, domain, coords);
    }

} //namespace gridtools

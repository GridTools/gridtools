#pragma once

namespace ico_operators {
    using backend_t = BACKEND;
    using icosahedral_topology_t = repository::icosahedral_topology_t;
    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_interval;
    typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;
}

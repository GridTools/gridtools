#pragma once
#include "stencil_composition/axis.hpp"
#include "common/halo_descriptor.hpp"
#include "stencil_composition/common_grids/grid_cg.hpp"
#include "stencil_composition/icosahedral_grids/icosahedral_topology.hpp"

namespace gridtools {

    template < typename Axis, typename GridTopology >
    struct grid : public grid_cg< Axis >, public clonable_to_gpu< grid< Axis, GridTopology > > {
        GRIDTOOLS_STATIC_ASSERT((is_interval< Axis >::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_grid_topology< GridTopology >::value), "Internal Error: wrong type");

        typedef GridTopology grid_topology_t;

      private:
        GridTopology m_grid_topology;

      public:
        GT_FUNCTION
        // TODO make grid const
        explicit grid(GridTopology &grid_topology, const array< uint_t, 5 > &i, const array< uint_t, 5 > &j)
            : grid_cg< Axis >(i, j), m_grid_topology(grid_topology) {}

        __device__ grid(grid const &other) : grid_cg< Axis >(other), m_grid_topology(other.m_grid_topology) {}

        GT_FUNCTION
        GridTopology const &grid_topology() const { return m_grid_topology; }
    };

    template < typename Grid >
    struct is_grid : boost::mpl::false_ {};

    template < typename Axis, typename GridTopology >
    struct is_grid< grid< Axis, GridTopology > > : boost::mpl::true_ {};
}

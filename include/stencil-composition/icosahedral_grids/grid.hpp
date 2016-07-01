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
#include "stencil-composition/axis.hpp"
#include "common/halo_descriptor.hpp"
#include "stencil-composition/common_grids/grid_cg.hpp"
#include "stencil-composition/icosahedral_grids/icosahedral_topology.hpp"

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

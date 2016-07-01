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
#include "unstructured_grid.hpp"

namespace gridtools {
    template <>
    std::list< array< uint_t, 4 > > const &
        unstructured_grid::neighbours_of< typename unstructured_grid::grid_topology_t::cells,
            typename unstructured_grid::grid_topology_t::cells >(array< uint_t, 4 > const &coords) {
        return m_cell_to_cells.at(coords);
    }

    template <>
    std::list< array< uint_t, 4 > > const &
        unstructured_grid::neighbours_of< typename unstructured_grid::grid_topology_t::edges,
            typename unstructured_grid::grid_topology_t::edges >(array< uint_t, 4 > const &coords) {
        return m_edge_to_edges.at(coords);
    }
    template <>
    std::list< array< uint_t, 4 > > const &
        unstructured_grid::neighbours_of< typename unstructured_grid::grid_topology_t::vertexes,
            typename unstructured_grid::grid_topology_t::vertexes >(array< uint_t, 4 > const &coords) {
        return m_vertex_to_vertexes.at(coords);
    }

    template <>
    std::list< array< uint_t, 4 > > const &
        unstructured_grid::neighbours_of< typename unstructured_grid::grid_topology_t::edges,
            typename unstructured_grid::grid_topology_t::cells >(array< uint_t, 4 > const &coords) {
        return m_edge_to_cells.at(coords);
    }

    template <>
    std::list< array< uint_t, 4 > > const &
        unstructured_grid::neighbours_of< typename unstructured_grid::grid_topology_t::cells,
            typename unstructured_grid::grid_topology_t::edges >(array< uint_t, 4 > const &coords) {
        return m_cell_to_edges.at(coords);
    }
}

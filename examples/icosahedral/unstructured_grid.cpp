/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#include "unstructured_grid.hpp"

namespace gridtools {
    template <>
    std::list<array<uint_t, 4>> const &
    unstructured_grid::neighbours_of<typename unstructured_grid::grid_topology_t::cells,
        typename unstructured_grid::grid_topology_t::cells>(array<uint_t, 4> const &coords) {
        return m_cell_to_cells.at(coords);
    }

    template <>
    std::list<array<uint_t, 4>> const &
    unstructured_grid::neighbours_of<typename unstructured_grid::grid_topology_t::cells,
        typename unstructured_grid::grid_topology_t::edges>(array<uint_t, 4> const &coords) {
        return m_cell_to_edges.at(coords);
    }

    template <>
    std::list<array<uint_t, 4>> const &
    unstructured_grid::neighbours_of<typename unstructured_grid::grid_topology_t::edges,
        typename unstructured_grid::grid_topology_t::edges>(array<uint_t, 4> const &coords) {
        return m_edge_to_edges.at(coords);
    }

    template <>
    std::list<array<uint_t, 4>> const &
    unstructured_grid::neighbours_of<typename unstructured_grid::grid_topology_t::edges,
        typename unstructured_grid::grid_topology_t::cells>(array<uint_t, 4> const &coords) {
        return m_edge_to_cells.at(coords);
    }

    template <>
    std::list<array<uint_t, 4>> const &
    unstructured_grid::neighbours_of<typename unstructured_grid::grid_topology_t::edges,
        typename unstructured_grid::grid_topology_t::vertices>(array<uint_t, 4> const &coords) {
        return m_edge_to_vertices.at(coords);
    }

    template <>
    std::list<array<uint_t, 4>> const &
    unstructured_grid::neighbours_of<typename unstructured_grid::grid_topology_t::vertices,
        typename unstructured_grid::grid_topology_t::vertices>(array<uint_t, 4> const &coords) {
        return m_vertex_to_vertices.at(coords);
    }

    template <>
    std::list<array<uint_t, 4>> const &
    unstructured_grid::neighbours_of<typename unstructured_grid::grid_topology_t::vertices,
        typename unstructured_grid::grid_topology_t::edges>(array<uint_t, 4> const &coords) {
        return m_vertex_to_edges.at(coords);
    }
} // namespace gridtools

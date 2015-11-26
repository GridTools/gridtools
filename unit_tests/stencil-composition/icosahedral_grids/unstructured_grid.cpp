#include "unstructured_grid.hpp"

namespace gridtools {
    template<>
    std::list<array<uint_t, 4>> const& unstructured_grid::neighbours_of<
    typename unstructured_grid::grid_topology_t::cells,
    typename unstructured_grid::grid_topology_t::cells>(array<uint_t, 4> const& coords)
    {
        return m_cell_to_cells.at(coords);
    }

    template<>
    std::list<array<uint_t, 4>> const& unstructured_grid::neighbours_of<
    typename unstructured_grid::grid_topology_t::edges,
    typename unstructured_grid::grid_topology_t::edges>(array<uint_t, 4> const& coords)
    {
        return m_edge_to_edges.at(coords);
    }
}

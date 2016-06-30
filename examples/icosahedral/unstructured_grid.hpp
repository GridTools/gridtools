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
#include "common/defs.hpp"
#include "common/array.hpp"
#include <list>
#include <vector>
#include <iostream>
#include <common/defs.hpp>
#include <stencil-composition/backend.hpp>

namespace gridtools {
    class neighbour_list {
      public:
        explicit neighbour_list(array< uint_t, 4 > &dims) {
            m_neigh_indexes.resize(dims[0] * dims[1] * dims[2] * dims[3]);
            m_strides = {1, dims[0], dims[0] * dims[1], dims[0] * dims[1] * dims[2]};
            m_size = m_strides[3] * dims[3];
        }

        uint_t index(array< uint_t, 4 > coord) { return coord * m_strides; }

        std::list< array< uint_t, 4 > > &at(const uint_t i, const uint_t c, const uint_t j, const uint_t k) {
            return at({i, c, j, k});
        }
        std::list< array< uint_t, 4 > > &at(array< uint_t, 4 > const &coord) {
            assert(index(coord) < m_size);
            return m_neigh_indexes[index(coord)];
        }

        void insert_neighbour(array< uint_t, 4 > const &coord, array< uint_t, 4 > neighbour) {
            at(coord).push_back(neighbour);
        }

      private:
        uint_t m_size;
        std::vector< std::list< array< uint_t, 4 > > > m_neigh_indexes;
        array< uint_t, 4 > m_strides;
    };

    class unstructured_grid {
        using backend_t = backend< enumtype::Host, enumtype::icosahedral, enumtype::Naive >;
        using grid_topology_t = icosahedral_topology< backend_t >;

        static const int ncolors = 2;

      private:
        array< uint_t, 4 > m_celldims;
        array< uint_t, 4 > m_edgedims;
        array< uint_t, 4 > m_vertexdims;
        neighbour_list m_cell_to_cells;
        neighbour_list m_cell_to_edges;
        neighbour_list m_cell_to_vertexes;
        neighbour_list m_edge_to_edges;
        neighbour_list m_edge_to_cells;
        neighbour_list m_vertex_to_vertexes;

      public:
        explicit unstructured_grid(uint_t i, uint_t j, uint_t k)
            : m_celldims{i, 2, j, k}, m_edgedims{i, 3, j, k}, m_vertexdims{i, 1, j + 1, k}, m_cell_to_cells(m_celldims),
              m_cell_to_edges(m_celldims), m_cell_to_vertexes(m_celldims), m_edge_to_edges(m_edgedims),
              m_edge_to_cells(m_edgedims), m_vertex_to_vertexes(m_vertexdims) {
            construct_grid();
        }

        void construct_grid() {
            for (uint_t k = 0; k < m_celldims[3]; ++k) {
                for (uint_t i = 1; i < m_celldims[0] - 1; ++i) {
                    for (uint_t j = 1; j < m_celldims[2] - 1; ++j) {
                        m_cell_to_cells.insert_neighbour({i, 0, j, k}, {i - 1, 1, j, k});
                        m_cell_to_cells.insert_neighbour({i, 0, j, k}, {i, 1, j, k});
                        m_cell_to_cells.insert_neighbour({i, 0, j, k}, {i, 1, j - 1, k});
                        m_cell_to_cells.insert_neighbour({i, 1, j, k}, {i + 1, 0, j, k});
                        m_cell_to_cells.insert_neighbour({i, 1, j, k}, {i, 0, j, k});
                        m_cell_to_cells.insert_neighbour({i, 1, j, k}, {i, 0, j + 1, k});

                        m_cell_to_edges.insert_neighbour({i, 0, j, k}, {i, 1, j, k});
                        m_cell_to_edges.insert_neighbour({i, 0, j, k}, {i, 2, j, k});
                        m_cell_to_edges.insert_neighbour({i, 0, j, k}, {i, 0, j, k});
                        m_cell_to_edges.insert_neighbour({i, 1, j, k}, {i + 1, 1, j, k});
                        m_cell_to_edges.insert_neighbour({i, 1, j, k}, {i, 2, j, k});
                        m_cell_to_edges.insert_neighbour({i, 1, j, k}, {i, 0, j + 1, k});
                    }
                }
            }
            for (uint_t k = 0; k < m_edgedims[3]; ++k) {
                for (uint_t i = 1; i < m_edgedims[0] - 1; ++i) {
                    for (uint_t j = 1; j < m_edgedims[2] - 1; ++j) {
                        m_edge_to_edges.insert_neighbour({i, 0, j, k}, {i, 2, j - 1, k});
                        m_edge_to_edges.insert_neighbour({i, 0, j, k}, {i, 1, j, k});
                        m_edge_to_edges.insert_neighbour({i, 0, j, k}, {i + 1, 1, j - 1, k});
                        m_edge_to_edges.insert_neighbour({i, 0, j, k}, {i, 2, j, k});

                        m_edge_to_edges.insert_neighbour({i, 1, j, k}, {i, 0, j, k});
                        m_edge_to_edges.insert_neighbour({i, 1, j, k}, {i - 1, 2, j, k});
                        m_edge_to_edges.insert_neighbour({i, 1, j, k}, {i - 1, 0, j + 1, k});
                        m_edge_to_edges.insert_neighbour({i, 1, j, k}, {i, 2, j, k});

                        m_edge_to_edges.insert_neighbour({i, 2, j, k}, {i, 0, j, k});
                        m_edge_to_edges.insert_neighbour({i, 2, j, k}, {i, 1, j, k});
                        m_edge_to_edges.insert_neighbour({i, 2, j, k}, {i + 1, 1, j, k});
                        m_edge_to_edges.insert_neighbour({i, 2, j, k}, {i, 0, j + 1, k});

                        m_edge_to_cells.insert_neighbour({i, 0, j, k}, {i, 1, j - 1, k});
                        m_edge_to_cells.insert_neighbour({i, 0, j, k}, {i, 0, j, k});

                        m_edge_to_cells.insert_neighbour({i, 1, j, k}, {i, 0, j, k});
                        m_edge_to_cells.insert_neighbour({i, 1, j, k}, {i - 1, 1, j, k});

                        m_edge_to_cells.insert_neighbour({i, 2, j, k}, {i, 0, j, k});
                        m_edge_to_cells.insert_neighbour({i, 2, j, k}, {i, 1, j, k});
                    }
                }
            }
            for (uint_t k = 0; k < m_vertexdims[3]; ++k) {
                for (uint_t i = 1; i < m_vertexdims[0] - 1; ++i) {
                    for (uint_t j = 1; j < m_vertexdims[2] - 1; ++j) {
                        m_vertex_to_vertexes.insert_neighbour({i, 0, j, k}, {i, 0, j - 1, k});
                        m_vertex_to_vertexes.insert_neighbour({i, 0, j, k}, {i, 0, j + 1, k});
                        m_vertex_to_vertexes.insert_neighbour({i, 0, j, k}, {i + 1, 0, j, k});
                        m_vertex_to_vertexes.insert_neighbour({i, 0, j, k}, {i - 1, 0, j, k});
                        m_vertex_to_vertexes.insert_neighbour({i, 0, j, k}, {i + 1, 0, j - 1, k});
                        m_vertex_to_vertexes.insert_neighbour({i, 0, j, k}, {i - 1, 0, j + 1, k});
                    }
                }
            }
        }

        template < typename LocationTypeFrom, typename LocationTypeTo >
        std::list< array< uint_t, 4 > > const &neighbours_of(array< uint_t, 4 > const &coords);
    };

} // namespace gridtools

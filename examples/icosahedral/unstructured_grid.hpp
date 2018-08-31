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
#pragma once
#include <gridtools/common/array.hpp>
#include <gridtools/common/array_dot_product.hpp>
#include <gridtools/common/defs.hpp>
#include <gridtools/stencil-composition/backend.hpp>
#include <iostream>
#include <list>
#include <vector>

namespace gridtools {
    class neighbour_list {
      public:
        explicit neighbour_list(array<uint_t, 4> &dims) {
            m_neigh_indexes.resize(dims[0] * dims[1] * dims[2] * dims[3]);
            m_strides = {1, dims[0], dims[0] * dims[1], dims[0] * dims[1] * dims[2]};
            m_size = m_strides[3] * dims[3];
        }

        uint_t index(array<uint_t, 4> coord) { return array_dot_product(coord, m_strides); }

        std::list<array<uint_t, 4>> &at(const uint_t i, const uint_t c, const uint_t j, const uint_t k) {
            return at({i, c, j, k});
        }
        std::list<array<uint_t, 4>> &at(array<uint_t, 4> const &coord) {
            assert(index(coord) < m_size);
            return m_neigh_indexes[index(coord)];
        }

        void insert_neighbour(array<uint_t, 4> const &coord, array<uint_t, 4> neighbour) {
            at(coord).push_back(std::move(neighbour));
        }

      private:
        uint_t m_size;
        std::vector<std::list<array<uint_t, 4>>> m_neigh_indexes;
        array<uint_t, 4> m_strides;
    };

    class unstructured_grid {
        using backend_t = backend<platform::x86, grid_type::icosahedral, strategy::naive>;
        using grid_topology_t = icosahedral_topology<backend_t>;

        static const int ncolors = 2;

      private:
        array<uint_t, 4> m_celldims;
        array<uint_t, 4> m_edgedims;
        array<uint_t, 4> m_vertexdims;
        neighbour_list m_cell_to_cells;
        neighbour_list m_cell_to_edges;
        neighbour_list m_cell_to_vertices;
        neighbour_list m_edge_to_edges;
        neighbour_list m_edge_to_cells;
        neighbour_list m_edge_to_vertices;
        neighbour_list m_vertex_to_vertices;
        neighbour_list m_vertex_to_edges;

      public:
        explicit unstructured_grid(uint_t i, uint_t j, uint_t k)
            : m_celldims{i, 2, j, k}, m_edgedims{i, 3, j, k}, m_vertexdims{i, 1, j + 1, k}, m_cell_to_cells(m_celldims),
              m_cell_to_edges(m_celldims), m_cell_to_vertices(m_celldims), m_edge_to_edges(m_edgedims),
              m_edge_to_cells(m_edgedims), m_edge_to_vertices(m_edgedims), m_vertex_to_vertices(m_vertexdims),
              m_vertex_to_edges(m_vertexdims) {
            construct_grid();
        }

        inline void construct_grid() {
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

                        m_edge_to_cells.insert_neighbour({i, 1, j, k}, {i - 1, 1, j, k});
                        m_edge_to_cells.insert_neighbour({i, 1, j, k}, {i, 0, j, k});

                        m_edge_to_cells.insert_neighbour({i, 2, j, k}, {i, 1, j, k});
                        m_edge_to_cells.insert_neighbour({i, 2, j, k}, {i, 0, j, k});

                        m_edge_to_vertices.insert_neighbour({i, 0, j, k}, {i + 1, 0, j, k});
                        m_edge_to_vertices.insert_neighbour({i, 0, j, k}, {i, 0, j, k});

                        m_edge_to_vertices.insert_neighbour({i, 1, j, k}, {i, 0, j, k});
                        m_edge_to_vertices.insert_neighbour({i, 1, j, k}, {i, 0, j + 1, k});

                        m_edge_to_vertices.insert_neighbour({i, 2, j, k}, {i, 0, j + 1, k});
                        m_edge_to_vertices.insert_neighbour({i, 2, j, k}, {i + 1, 0, j, k});
                    }
                }
            }
            for (uint_t k = 0; k < m_vertexdims[3]; ++k) {
                for (uint_t i = 1; i < m_vertexdims[0] - 1; ++i) {
                    for (uint_t j = 1; j < m_vertexdims[2] - 1; ++j) {
                        m_vertex_to_vertices.insert_neighbour({i, 0, j, k}, {i, 0, j - 1, k});
                        m_vertex_to_vertices.insert_neighbour({i, 0, j, k}, {i, 0, j + 1, k});
                        m_vertex_to_vertices.insert_neighbour({i, 0, j, k}, {i + 1, 0, j, k});
                        m_vertex_to_vertices.insert_neighbour({i, 0, j, k}, {i - 1, 0, j, k});
                        m_vertex_to_vertices.insert_neighbour({i, 0, j, k}, {i + 1, 0, j - 1, k});
                        m_vertex_to_vertices.insert_neighbour({i, 0, j, k}, {i - 1, 0, j + 1, k});
                    }
                }
            }

            for (uint_t k = 0; k < m_vertexdims[3]; ++k) {
                for (uint_t i = 1; i < m_vertexdims[0] - 1; ++i) {
                    for (uint_t j = 1; j < m_vertexdims[2] - 1; ++j) {
                        m_vertex_to_edges.insert_neighbour({i, 0, j, k}, {i, 1, j - 1, k});
                        m_vertex_to_edges.insert_neighbour({i, 0, j, k}, {i - 1, 0, j, k});
                        m_vertex_to_edges.insert_neighbour({i, 0, j, k}, {i - 1, 2, j, k});
                        m_vertex_to_edges.insert_neighbour({i, 0, j, k}, {i, 1, j, k});
                        m_vertex_to_edges.insert_neighbour({i, 0, j, k}, {i, 0, j, k});
                        m_vertex_to_edges.insert_neighbour({i, 0, j, k}, {i, 2, j - 1, k});
                    }
                }
            }
        }

        template <typename LocationTypeFrom, typename LocationTypeTo>
        std::list<array<uint_t, 4>> const &neighbours_of(array<uint_t, 4> const &coords);
    };

} // namespace gridtools

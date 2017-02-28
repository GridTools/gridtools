/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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

#include "../icosahedral/unstructured_grid.hpp"
#include "operator_defs.hpp"
#include <random>

namespace ico_operators {
    using namespace gridtools;
    using namespace enumtype;
    using namespace expressions;

    using backend_t = BACKEND;

    class repository {
      public:
        using cell_storage_type = icosahedral_topology_t::storage_t< icosahedral_topology_t::cells, double >;
        using edge_storage_type = icosahedral_topology_t::storage_t< icosahedral_topology_t::edges, double >;
        using vertex_storage_type = icosahedral_topology_t::storage_t< icosahedral_topology_t::vertices, double >;

        using cell_2d_storage_type =
            icosahedral_topology_t::storage_t< icosahedral_topology_t::cells, double, selector< 1, 1, 1, -1 > >;
        using edge_2d_storage_type =
            icosahedral_topology_t::storage_t< icosahedral_topology_t::edges, double, selector< 1, 1, 1, -1 > >;
        using vertex_2d_storage_type =
            icosahedral_topology_t::storage_t< icosahedral_topology_t::vertices, double, selector< 1, 1, 1, -1 > >;

        using tmp_edge_storage_type =
            typename icosahedral_topology_t::temporary_storage_t< icosahedral_topology_t::edges, double >;
        using tmp_vertex_storage_type =
            typename icosahedral_topology_t::temporary_storage_t< icosahedral_topology_t::vertices, double >;
        using tmp_cell_storage_type =
            typename icosahedral_topology_t::temporary_storage_t< icosahedral_topology_t::cells, double >;

        using vertices_4d_storage_type =
            icosahedral_topology_t::storage_t< icosahedral_topology_t::vertices, double, selector< 1, 1, 1, 1, 1 > >;
        using cells_4d_storage_type =
            icosahedral_topology_t::storage_t< icosahedral_topology_t::cells, double, selector< 1, 1, 1, 1, 1 > >;
        using edges_4d_storage_type =
            icosahedral_topology_t::storage_t< icosahedral_topology_t::edges, double, selector< 1, 1, 1, 1, 1 > >;
        using edges_of_cells_storage_type =
            icosahedral_topology_t::storage_t< icosahedral_topology_t::cells, double, selector< 1, 1, 1, -1, 1 > >;
        using edges_of_vertices_storage_type =
            icosahedral_topology_t::storage_t< icosahedral_topology_t::vertices, double, selector< 1, 1, 1, -1, 1 > >;

      private:
        icosahedral_topology_t icosahedral_grid_;

        edge_storage_type m_u;
        vertex_storage_type m_out_vertex;
        vertex_storage_type m_curl_u_ref;
        edge_storage_type m_grad_n_ref;
        cell_storage_type m_div_u_ref;
        edge_storage_type m_lap_ref;
        vertex_2d_storage_type m_dual_area;
        vertex_2d_storage_type m_dual_area_reciprocal;
        cell_2d_storage_type m_cell_area;
        cell_2d_storage_type m_cell_area_reciprocal;

      private:
        unstructured_grid m_ugrid;
        cells_4d_storage_type m_div_weights;
        edge_2d_storage_type m_edge_length;
        edge_2d_storage_type m_edge_length_reciprocal;

        edge_2d_storage_type m_dual_edge_length;
        edge_2d_storage_type m_dual_edge_length_reciprocal;
        edges_of_vertices_storage_type m_edge_orientation;
        edges_of_cells_storage_type m_orientation_of_normal;

      public:
        uint_t idim() { return m_idim; }
        uint_t jdim() { return m_jdim; }
        uint_t kdim() { return m_kdim; }

        repository(const uint_t idim, const uint_t jdim, const uint_t kdim)
            : icosahedral_grid_(idim, jdim, kdim), m_idim(idim), m_jdim(jdim), m_kdim(kdim), m_ugrid(idim, jdim, kdim),
              m_u(icosahedral_grid_.make_storage< icosahedral_topology_t::edges, double >("u")),
              m_out_vertex(icosahedral_grid_.make_storage< icosahedral_topology_t::vertices, double >("out_vertex")),
              m_curl_u_ref(icosahedral_grid_.make_storage< icosahedral_topology_t::vertices, double >("curl_u_ref")),
              m_grad_n_ref(icosahedral_grid_.make_storage< icosahedral_topology_t::edges, double >("grad_n_ref")),
              m_div_u_ref(icosahedral_grid_.make_storage< icosahedral_topology_t::cells, double >("div_u_ref")),
              m_lap_ref(icosahedral_grid_.make_storage< icosahedral_topology_t::edges, double >("lap_ref")),
              m_dual_area(
                  icosahedral_grid_.make_storage< icosahedral_topology_t::vertices, double, selector< 1, 1, 1, -1 > >(
                      "dual_area")),
              m_cell_area(
                  icosahedral_grid_.make_storage< icosahedral_topology_t::cells, double, selector< 1, 1, 1, -1 > >(
                      "cell_area")),
              m_cell_area_reciprocal(
                  icosahedral_grid_.make_storage< icosahedral_topology_t::cells, double, selector< 1, 1, 1, -1 > >(
                      "cell_area_reciprocal")),
              m_edge_length(
                  icosahedral_grid_.make_storage< icosahedral_topology_t::edges, double, selector< 1, 1, 1, -1 > >(
                      "edge_length")),
              m_edge_length_reciprocal(
                  icosahedral_grid_.make_storage< icosahedral_topology_t::edges, double, selector< 1, 1, 1, -1 > >(
                      "edge_length_reciprocal")),
              m_dual_edge_length(
                  icosahedral_grid_.make_storage< icosahedral_topology_t::edges, double, selector< 1, 1, 1, -1 > >(
                      "dual_edge_length")),
              m_dual_edge_length_reciprocal(
                  icosahedral_grid_.make_storage< icosahedral_topology_t::edges, double, selector< 1, 1, 1, -1 > >(
                      "dual_edge_length_reciprocal")),
              m_edge_orientation(
                  icosahedral_grid_
                      .make_storage< icosahedral_topology_t::vertices, double, selector< 1, 1, 1, -1, 1 > >(
                          "edge_orientation", 6)),
              m_orientation_of_normal(
                  icosahedral_grid_.make_storage< icosahedral_topology_t::cells, double, selector< 1, 1, 1, -1, 1 > >(
                      "orientation_of_normal", 3)),
              m_div_weights(
                  icosahedral_grid_.make_storage< icosahedral_topology_t::cells, double, selector< 1, 1, 1, 1, 1 > >(
                      "div_weights", 3)),
              m_dual_area_reciprocal(
                  icosahedral_grid_.make_storage< icosahedral_topology_t::vertices, double, selector< 1, 1, 1, -1 > >(
                      "dual_area_reciprocal")) {}

        void init_fields() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0, 1);
            const double PI = std::atan(1.) * 4.;

            double dx = 1. / (double)(m_idim);
            double dy = 1. / (double)(m_jdim);

            // init u
            for (int i = 0; i < m_idim; ++i) {
                for (int c = 0; c < icosahedral_topology_t::edges::n_colors::value; ++c) {
                    for (int j = 0; j < m_jdim; ++j) {
                        double x = dx * (i + c / (double)icosahedral_topology_t::edges::n_colors::value);
                        double y = dy * j;

                        for (uint_t k = 0; k < m_kdim; ++k) {
                            m_u(i, c, j, k) = k + 8 * (2. + cos(PI * (x + 1.5 * y)) + sin(2 * PI * (x + 1.5 * y))) / 4.;
                            m_edge_length(i, c, j, 0) =
                                2.95 + (2. + cos(PI * (x + 1.5 * y)) + sin(2 * PI * (x + 1.5 * y))) / 4.;
                            m_edge_length_reciprocal(i, c, j, 0) = 1 / m_edge_length(i, c, j, 0);

                            m_dual_edge_length(i, c, j, 0) =
                                2.2 + (2. + cos(PI * (x + 2.5 * y)) + sin(2 * PI * (x + 3.5 * y))) / 4.;
                            m_dual_edge_length_reciprocal(i, c, j, 0) = 1 / m_dual_edge_length(i, c, j, 0);
                        }
                    }
                }
            }

            for (int i = 0; i < m_idim; ++i) {
                for (int c = 0; c < icosahedral_topology_t::vertices::n_colors::value; ++c) {
                    for (int j = 0; j < m_jdim; ++j) {
                        double x = dx * (i + c / icosahedral_topology_t::vertices::n_colors::value);
                        double y = dy * j;

                        m_dual_area(i, c, j, 0) =
                            1.1 + (2. + cos(PI * (1.5 * x + y)) + sin(1.5 * PI * (x + 1.5 * y))) / 4.;
                    }
                }
            }
            for (int i = 0; i < m_idim; ++i) {
                for (int c = 0; c < icosahedral_topology_t::cells::n_colors::value; ++c) {
                    for (int j = 0; j < m_jdim; ++j) {
                        double x = dx * (i + c / icosahedral_topology_t::cells::n_colors::value);
                        double y = dy * j;

                        m_cell_area(i, c, j, 0) =
                            2.53 + (2. + cos(PI * (1.5 * x + 2.5 * y)) + sin(2 * PI * (x + 1.5 * y))) / 4.;
                        m_cell_area_reciprocal(i, c, j, 0) = (float_type)1. / m_cell_area(i, c, j, 0);
                    }
                }
            }

            // dual_area_reciprocal_
            for (int i = 0; i < m_idim; ++i) {
                for (int c = 0; c < icosahedral_topology_t::vertices::n_colors::value; ++c) {
                    for (int j = 0; j < icosahedral_grid_.m_dims[1]; ++j) {
                        m_dual_area_reciprocal(i, c, j, 0) = 1. / m_dual_area(i, c, j, 0);
                    }
                }
            }

            // orientation of normal
            for (int i = 0; i < m_idim; ++i) {
                for (int j = 0; j < m_jdim; ++j) {
                    for (uint_t e = 0; e < 3; ++e) {
                        m_orientation_of_normal(i, 0, j, 0, e) = 1;
                        m_orientation_of_normal(i, 1, j, 0, e) = -1;
                    }
                }
            }

            // edge orientation
            for (int i = 0; i < m_idim; ++i) {
                for (int c = 0; c < icosahedral_topology_t::vertices::n_colors::value; ++c) {
                    for (int j = 0; j < m_jdim; ++j) {
                        for (uint_t k = 0; k < m_kdim; ++k) {
                            m_edge_orientation(i, c, j, 0, 0) = -1;
                            m_edge_orientation(i, c, j, 0, 2) = -1;
                            m_edge_orientation(i, c, j, 0, 4) = -1;
                            m_edge_orientation(i, c, j, 0, 1) = 1;
                            m_edge_orientation(i, c, j, 0, 3) = 1;
                            m_edge_orientation(i, c, j, 0, 5) = 1;
                        }
                    }
                }
            }

            m_curl_u_ref.initialize(0.0);
            m_div_u_ref.initialize(0.0);
            m_lap_ref.initialize(0.0);
            m_div_weights.initialize(0.0);
            m_grad_n_ref.initialize(0.0);
        }

        inline void generate_div_ref() {
            for (uint_t i = halo_nc; i < m_idim - halo_nc; ++i) {
                for (uint_t c = 0; c < icosahedral_topology_t::cells::n_colors::value; ++c) {
                    for (uint_t j = halo_mc; j < m_jdim - halo_mc; ++j) {
                        for (uint_t k = 0; k < m_kdim; ++k) {
                            auto neighbours =
                                m_ugrid.neighbours_of< icosahedral_topology_t::cells, icosahedral_topology_t::edges >(
                                    {i, c, j, k});
                            ushort_t e = 0;
                            for (auto iter = neighbours.begin(); iter != neighbours.end(); ++iter) {
                                m_div_u_ref(i, c, j, k) +=
                                    m_orientation_of_normal(i, c, j, k, e) * m_u(*iter) * m_edge_length(*iter);
                                ++e;
                            }
                            m_div_u_ref(i, c, j, k) /= m_cell_area(i, c, j, k);
                        }
                    }
                }
            }
        }

        inline void generate_curl_ref() {
            // curl_u_ref_
            for (uint_t i = halo_nc; i < m_idim - halo_nc; ++i) {
                for (uint_t c = 0; c < icosahedral_topology_t::vertices::n_colors::value; ++c) {
                    for (uint_t j = halo_mc; j < m_jdim - halo_mc; ++j) {
                        for (uint_t k = 0; k < m_kdim; ++k) {
                            auto neighbours = m_ugrid.neighbours_of< icosahedral_topology_t::vertices,
                                icosahedral_topology_t::edges >({i, c, j, k});
                            ushort_t e = 0;
                            for (auto iter = neighbours.begin(); iter != neighbours.end(); ++iter) {
                                m_curl_u_ref(i, c, j, k) +=
                                    m_edge_orientation(i, c, j, 0, e) * m_u(*iter) * m_dual_edge_length(*iter);
                                ++e;
                            }
                            m_curl_u_ref(i, c, j, k) /= m_dual_area(i, c, j, 0);
                        }
                    }
                }
            }
        }

        inline void generate_lap_ref() {

            generate_curl_ref();
            generate_div_ref();

            for (uint_t i = halo_nc; i < m_idim - halo_nc; ++i) {
                for (uint_t c = 0; c < icosahedral_topology_t::edges::n_colors::value; ++c) {
                    for (uint_t j = halo_mc; j < m_jdim - halo_mc; ++j) {
                        for (uint_t k = 0; k < m_kdim; ++k) {
                            auto neighbours_ec =
                                m_ugrid.neighbours_of< icosahedral_topology_t::edges, icosahedral_topology_t::cells >(
                                    {i, c, j, k});

                            auto neighbours_vc = m_ugrid.neighbours_of< icosahedral_topology_t::edges,
                                icosahedral_topology_t::vertices >({i, c, j, k});

                            double grad_n = 0;
                            double grad_tau = 0;
                            ushort_t e = 0;
                            assert(neighbours_ec.size() == 2);
                            assert(neighbours_vc.size() == 2);
                            for (auto iter = neighbours_ec.begin(); iter != neighbours_ec.end(); ++iter) {
                                if (e == 1) {
                                    grad_n += m_div_u_ref(*iter);
                                } else {
                                    grad_n -= m_div_u_ref(*iter);
                                }

                                ++e;
                            }

                            e = 0;
                            for (auto iter = neighbours_vc.begin(); iter != neighbours_vc.end(); ++iter) {
                                if (e == 1) {
                                    grad_tau += m_curl_u_ref(*iter);
                                } else {
                                    grad_tau -= m_curl_u_ref(*iter);
                                }

                                ++e;
                            }

                            grad_n *= m_dual_edge_length_reciprocal(i, c, j, k);
                            grad_tau *= m_edge_length_reciprocal(i, c, j, k);

                            m_lap_ref(i, c, j, k) = grad_n - grad_tau;
                        }
                    }
                }
            }
        }

        icosahedral_topology_t &icosahedral_grid() { return icosahedral_grid_; }

        edge_storage_type &u() { return m_u; }
        vertex_storage_type &out_vertex() { return m_out_vertex; }
        vertex_storage_type &curl_u_ref() { return m_curl_u_ref; }
        cell_storage_type &div_u_ref() { return m_div_u_ref; }
        edge_storage_type &lap_ref() { return m_lap_ref; }
        vertex_2d_storage_type &dual_area() { return m_dual_area; }
        cell_2d_storage_type &cell_area() { return m_cell_area; }
        cell_2d_storage_type &cell_area_reciprocal() { return m_cell_area_reciprocal; }
        vertex_2d_storage_type &dual_area_reciprocal() { return m_dual_area_reciprocal; }

        edge_2d_storage_type &edge_length() { return m_edge_length; }
        edge_2d_storage_type &edge_length_reciprocal() { return m_edge_length_reciprocal; }
        edge_2d_storage_type &dual_edge_length() { return m_dual_edge_length; }
        edge_2d_storage_type &dual_edge_length_reciprocal() { return m_dual_edge_length_reciprocal; }
        cells_4d_storage_type &div_weights() { return m_div_weights; }

        edges_of_vertices_storage_type &edge_orientation() { return m_edge_orientation; }
        edges_of_cells_storage_type &orientation_of_normal() { return m_orientation_of_normal; }

      public:
        const uint_t m_idim, m_jdim, m_kdim;
        const uint_t halo_nc = 2;
        const uint_t halo_mc = 2;
        const uint_t halo_k = 0;
    };
}

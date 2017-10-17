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
        static constexpr uint_t halo_nc = 2;
        static constexpr uint_t halo_mc = 2;
        static constexpr uint_t halo_k = 0;

        using halo_t = halo< 2, 0, 2, 0 >;
        using halo_5d_t = halo< 2, 0, 2, 0, 0 >;
        using cell_storage_type =
            icosahedral_topology_t::storage_t< icosahedral_topology_t::cells, float_type, halo_t >;
        using edge_storage_type =
            icosahedral_topology_t::storage_t< icosahedral_topology_t::edges, float_type, halo_t >;
        using vertex_storage_type =
            icosahedral_topology_t::storage_t< icosahedral_topology_t::vertices, float_type, halo_t >;

        using cell_2d_storage_type = icosahedral_topology_t::
            storage_t< icosahedral_topology_t::cells, float_type, halo_t, selector< 1, 1, 1, 0 > >;
        using edge_2d_storage_type = icosahedral_topology_t::
            storage_t< icosahedral_topology_t::edges, float_type, halo_t, selector< 1, 1, 1, 0 > >;
        using vertex_2d_storage_type = icosahedral_topology_t::
            storage_t< icosahedral_topology_t::vertices, float_type, halo_t, selector< 1, 1, 1, 0 > >;

        using vertices_4d_storage_type = icosahedral_topology_t::
            storage_t< icosahedral_topology_t::vertices, float_type, halo_5d_t, selector< 1, 1, 1, 1, 1 > >;
        using cells_4d_storage_type = icosahedral_topology_t::
            storage_t< icosahedral_topology_t::cells, float_type, halo_5d_t, selector< 1, 1, 1, 1, 1 > >;
        using edges_4d_storage_type = icosahedral_topology_t::
            storage_t< icosahedral_topology_t::edges, float_type, halo_5d_t, selector< 1, 1, 1, 1, 1 > >;
        using edges_of_cells_storage_type = icosahedral_topology_t::
            storage_t< icosahedral_topology_t::cells, float_type, halo_5d_t, selector< 1, 1, 1, 0, 1 > >;
        using edges_of_vertices_storage_type = icosahedral_topology_t::
            storage_t< icosahedral_topology_t::vertices, float_type, halo_5d_t, selector< 1, 1, 1, 0, 1 > >;

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
              m_u(icosahedral_grid_.make_storage< icosahedral_topology_t::edges, float_type, halo_t >("u")),
              m_out_vertex(
                  icosahedral_grid_.make_storage< icosahedral_topology_t::vertices, float_type, halo_t >("out_vertex")),
              m_curl_u_ref(
                  icosahedral_grid_.make_storage< icosahedral_topology_t::vertices, float_type, halo_t >("curl_u_ref")),
              m_grad_n_ref(
                  icosahedral_grid_.make_storage< icosahedral_topology_t::edges, float_type, halo_t >("grad_n_ref")),
              m_div_u_ref(
                  icosahedral_grid_.make_storage< icosahedral_topology_t::cells, float_type, halo_t >("div_u_ref")),
              m_lap_ref(icosahedral_grid_.make_storage< icosahedral_topology_t::edges, float_type, halo_t >("lap_ref")),
              m_dual_area(
                  icosahedral_grid_
                      .make_storage< icosahedral_topology_t::vertices, float_type, halo_t, selector< 1, 1, 1, 0 > >(
                          "dual_area")),
              m_cell_area(
                  icosahedral_grid_
                      .make_storage< icosahedral_topology_t::cells, float_type, halo_t, selector< 1, 1, 1, 0 > >(
                          "cell_area")),
              m_cell_area_reciprocal(
                  icosahedral_grid_
                      .make_storage< icosahedral_topology_t::cells, float_type, halo_t, selector< 1, 1, 1, 0 > >(
                          "cell_area_reciprocal")),
              m_edge_length(
                  icosahedral_grid_
                      .make_storage< icosahedral_topology_t::edges, float_type, halo_t, selector< 1, 1, 1, 0 > >(
                          "edge_length")),
              m_edge_length_reciprocal(
                  icosahedral_grid_
                      .make_storage< icosahedral_topology_t::edges, float_type, halo_t, selector< 1, 1, 1, 0 > >(
                          "edge_length_reciprocal")),
              m_dual_edge_length(
                  icosahedral_grid_
                      .make_storage< icosahedral_topology_t::edges, float_type, halo_t, selector< 1, 1, 1, 0 > >(
                          "dual_edge_length")),
              m_dual_edge_length_reciprocal(
                  icosahedral_grid_
                      .make_storage< icosahedral_topology_t::edges, float_type, halo_t, selector< 1, 1, 1, 0 > >(
                          "dual_edge_length_reciprocal")),
              m_edge_orientation(icosahedral_grid_.make_storage< icosahedral_topology_t::vertices,
                                 float_type,
                                 halo_5d_t,
                                 selector< 1, 1, 1, 0, 1 > >("edge_orientation", 6)),
              m_orientation_of_normal(
                  icosahedral_grid_
                      .make_storage< icosahedral_topology_t::cells, float_type, halo_5d_t, selector< 1, 1, 1, 0, 1 > >(
                          "orientation_of_normal", 3)),
              m_div_weights(
                  icosahedral_grid_
                      .make_storage< icosahedral_topology_t::cells, float_type, halo_5d_t, selector< 1, 1, 1, 1, 1 > >(
                          "div_weights", 3)),
              m_dual_area_reciprocal(
                  icosahedral_grid_
                      .make_storage< icosahedral_topology_t::vertices, float_type, halo_t, selector< 1, 1, 1, 0 > >(
                          "dual_area_reciprocal")) {}

        void init_fields() {
            const float_type PI = std::atan(1.) * 4.;

            float_type dx = 1. / (float_type)(m_idim);
            float_type dy = 1. / (float_type)(m_jdim);

            // create views
            auto m_u_v = make_host_view(m_u);
            auto m_edge_length_v = make_host_view(m_edge_length);
            auto m_edge_length_reciprocal_v = make_host_view(m_edge_length_reciprocal);
            auto m_dual_edge_length_v = make_host_view(m_dual_edge_length);
            auto m_dual_edge_length_reciprocal_v = make_host_view(m_dual_edge_length_reciprocal);
            auto m_dual_area_v = make_host_view(m_dual_area);
            auto m_cell_area_v = make_host_view(m_cell_area);
            auto m_cell_area_reciprocal_v = make_host_view(m_cell_area_reciprocal);
            auto m_dual_area_reciprocal_v = make_host_view(m_dual_area_reciprocal);
            auto m_orientation_of_normal_v = make_host_view(m_orientation_of_normal);
            auto m_edge_orientation_v = make_host_view(m_edge_orientation);

            // init u
            for (int i = 0; i < m_idim; ++i) {
                for (int c = 0; c < icosahedral_topology_t::edges::n_colors::value; ++c) {
                    for (int j = 0; j < m_jdim; ++j) {
                        float_type x = dx * (i + c / (float_type)icosahedral_topology_t::edges::n_colors::value);
                        float_type y = dy * j;

                        for (uint_t k = 0; k < m_kdim; ++k) {
                            m_u_v(i, c, j, k) = k +
                                                (float_type)8 *
                                                    ((float_type)2. + cos(PI * (x + (float_type)1.5 * y)) +
                                                        sin((float_type)2 * PI * (x + (float_type)1.5 * y))) /
                                                    (float_type)4.;
                        }
                        m_edge_length_v(i, c, j, 0) = (float_type)2.95 +
                                                      ((float_type)2. + cos(PI * (x + (float_type)1.5 * y)) +
                                                          sin((float_type)2 * PI * (x + (float_type)1.5 * y))) /
                                                          (float_type)4.;
                        m_edge_length_reciprocal_v(i, c, j, 0) = (float_type)1 / m_edge_length_v(i, c, j, 0);
                        m_dual_edge_length_v(i, c, j, 0) = (float_type)2.2 +
                                                           ((float_type)2. + cos(PI * (x + (float_type)2.5 * y)) +
                                                               sin((float_type)2 * PI * (x + (float_type)3.5 * y))) /
                                                               (float_type)4.;
                        m_dual_edge_length_reciprocal_v(i, c, j, 0) = (float_type)1 / m_dual_edge_length_v(i, c, j, 0);
                    }
                }
            }

            for (int i = 0; i < m_idim; ++i) {
                for (int c = 0; c < icosahedral_topology_t::vertices::n_colors::value; ++c) {
                    for (int j = 0; j < m_jdim; ++j) {
                        float_type x = dx * (i + c / (float_type)icosahedral_topology_t::vertices::n_colors::value);
                        float_type y = dy * j;

                        m_dual_area_v(i, c, j, 0) = (float_type)1.1 +
                                                    ((float_type)2. + cos(PI * ((float_type)1.5 * x + y)) +
                                                        sin((float_type)1.5 * PI * (x + (float_type)1.5 * y))) /
                                                        (float_type)4.;
                    }
                }
            }
            for (int i = 0; i < m_idim; ++i) {
                for (int c = 0; c < icosahedral_topology_t::cells::n_colors::value; ++c) {
                    for (int j = 0; j < m_jdim; ++j) {
                        float_type x = dx * (i + c / (float_type)icosahedral_topology_t::cells::n_colors::value);
                        float_type y = dy * j;

                        m_cell_area_v(i, c, j, 0) =
                            (float_type)2.53 +
                            ((float_type)2. + cos(PI * ((float_type)1.5 * x + (float_type)2.5 * y)) +
                                sin((float_type)2 * PI * (x + (float_type)1.5 * y))) /
                                (float_type)4.;
                        m_cell_area_reciprocal_v(i, c, j, 0) = (float_type)1. / m_cell_area_v(i, c, j, 0);
                    }
                }
            }

            // dual_area_reciprocal_
            for (int i = 0; i < m_idim; ++i) {
                for (int c = 0; c < icosahedral_topology_t::vertices::n_colors::value; ++c) {
                    for (int j = 0; j < icosahedral_grid_.m_dims[1]; ++j) {
                        m_dual_area_reciprocal_v(i, c, j, 0) = (float_type)1. / m_dual_area_v(i, c, j, 0);
                    }
                }
            }

            // orientation of normal
            for (int i = 0; i < m_idim; ++i) {
                for (int j = 0; j < m_jdim; ++j) {
                    for (uint_t e = 0; e < 3; ++e) {
                        m_orientation_of_normal_v(i, 0, j, 0, e) = 1;
                        m_orientation_of_normal_v(i, 1, j, 0, e) = -1;
                    }
                }
            }

            // edge orientation
            for (int i = 0; i < m_idim; ++i) {
                for (int c = 0; c < icosahedral_topology_t::vertices::n_colors::value; ++c) {
                    for (int j = 0; j < m_jdim; ++j) {
                        m_edge_orientation_v(i, c, j, 0, 0) = -1;
                        m_edge_orientation_v(i, c, j, 0, 2) = -1;
                        m_edge_orientation_v(i, c, j, 0, 4) = -1;
                        m_edge_orientation_v(i, c, j, 0, 1) = 1;
                        m_edge_orientation_v(i, c, j, 0, 3) = 1;
                        m_edge_orientation_v(i, c, j, 0, 5) = 1;
                    }
                }
            }
            // reinitialize some fields to 0.0
            m_curl_u_ref = vertex_storage_type(*m_curl_u_ref.get_storage_info_ptr(), 0.0);
            m_div_u_ref = cell_storage_type(*m_div_u_ref.get_storage_info_ptr(), 0.0);
            m_lap_ref = edge_storage_type(*m_lap_ref.get_storage_info_ptr(), 0.0);
            m_div_weights = cells_4d_storage_type(*m_div_weights.get_storage_info_ptr(), 0.0);
            m_grad_n_ref = edge_storage_type(*m_grad_n_ref.get_storage_info_ptr(), 0.0);
        }

        inline void generate_div_ref() {
            // create views
            auto m_div_u_ref_v = make_host_view(m_div_u_ref);
            auto m_orientation_of_normal_v = make_host_view(m_orientation_of_normal);
            auto m_edge_length_v = make_host_view(m_edge_length);
            auto m_u_v = make_host_view(m_u);
            auto m_cell_area_v = make_host_view(m_cell_area);

            for (uint_t i = halo_nc; i < m_idim - halo_nc; ++i) {
                for (uint_t c = 0; c < icosahedral_topology_t::cells::n_colors::value; ++c) {
                    for (uint_t j = halo_mc; j < m_jdim - halo_mc; ++j) {
                        for (uint_t k = 0; k < m_kdim; ++k) {
                            auto neighbours =
                                m_ugrid.neighbours_of< icosahedral_topology_t::cells, icosahedral_topology_t::edges >(
                                    {i, c, j, k});
                            ushort_t e = 0;
                            for (auto iter = neighbours.begin(); iter != neighbours.end(); ++iter) {
                                m_div_u_ref_v(i, c, j, k) +=
                                    m_orientation_of_normal_v(i, c, j, k, e) *
                                    m_u_v((*iter)[0], (*iter)[1], (*iter)[2], (*iter)[3]) *
                                    m_edge_length_v((*iter)[0], (*iter)[1], (*iter)[2], (*iter)[3]);
                                ++e;
                            }
                            m_div_u_ref_v(i, c, j, k) /= m_cell_area_v(i, c, j, k);
                        }
                    }
                }
            }
        }

        inline void generate_curl_ref() {
            // curl_u_ref_
            // create views
            auto m_curl_u_ref_v = make_host_view(m_curl_u_ref);
            auto m_edge_orientation_v = make_host_view(m_edge_orientation);
            auto m_u_v = make_host_view(m_u);
            auto m_dual_edge_length_v = make_host_view(m_dual_edge_length);
            auto m_edge_length_v = make_host_view(m_edge_length);
            auto m_dual_area_v = make_host_view(m_dual_area);

            for (uint_t i = halo_nc; i < m_idim - halo_nc; ++i) {
                for (uint_t c = 0; c < icosahedral_topology_t::vertices::n_colors::value; ++c) {
                    for (uint_t j = halo_mc; j < m_jdim - halo_mc; ++j) {
                        for (uint_t k = 0; k < m_kdim; ++k) {
                            auto neighbours = m_ugrid.neighbours_of< icosahedral_topology_t::vertices,
                                icosahedral_topology_t::edges >({i, c, j, k});
                            ushort_t e = 0;
                            for (auto iter = neighbours.begin(); iter != neighbours.end(); ++iter) {
                                m_curl_u_ref_v(i, c, j, k) +=
                                    m_edge_orientation_v(i, c, j, 0, e) *
                                    m_u_v((*iter)[0], (*iter)[1], (*iter)[2], (*iter)[3]) *
                                    m_dual_edge_length_v((*iter)[0], (*iter)[1], (*iter)[2], (*iter)[3]);
                                ++e;
                            }
                            m_curl_u_ref_v(i, c, j, k) /= m_dual_area_v(i, c, j, 0);
                        }
                    }
                }
            }
        }

        inline void generate_lap_ref() {

            generate_curl_ref();
            generate_div_ref();
            auto m_div_u_ref_v = make_host_view(m_div_u_ref);
            auto m_curl_u_ref_v = make_host_view(m_curl_u_ref);
            auto m_dual_edge_length_reciprocal_v = make_host_view(m_dual_edge_length_reciprocal);
            auto m_edge_length_reciprocal_v = make_host_view(m_edge_length_reciprocal);
            auto m_lap_ref_v = make_host_view(m_lap_ref);

            for (uint_t i = halo_nc; i < m_idim - halo_nc; ++i) {
                for (uint_t c = 0; c < icosahedral_topology_t::edges::n_colors::value; ++c) {
                    for (uint_t j = halo_mc; j < m_jdim - halo_mc; ++j) {
                        for (uint_t k = 0; k < m_kdim; ++k) {
                            auto neighbours_ec =
                                m_ugrid.neighbours_of< icosahedral_topology_t::edges, icosahedral_topology_t::cells >(
                                    {i, c, j, k});

                            auto neighbours_vc = m_ugrid.neighbours_of< icosahedral_topology_t::edges,
                                icosahedral_topology_t::vertices >({i, c, j, k});

                            float_type grad_n = 0;
                            float_type grad_tau = 0;
                            ushort_t e = 0;
                            assert(neighbours_ec.size() == 2);
                            assert(neighbours_vc.size() == 2);
                            for (auto iter = neighbours_ec.begin(); iter != neighbours_ec.end(); ++iter) {
                                if (e == 1) {
                                    grad_n += m_div_u_ref_v((*iter)[0], (*iter)[1], (*iter)[2], (*iter)[3]);
                                } else {
                                    grad_n -= m_div_u_ref_v((*iter)[0], (*iter)[1], (*iter)[2], (*iter)[3]);
                                }

                                ++e;
                            }

                            e = 0;
                            for (auto iter = neighbours_vc.begin(); iter != neighbours_vc.end(); ++iter) {
                                if (e == 1) {
                                    grad_tau += m_curl_u_ref_v((*iter)[0], (*iter)[1], (*iter)[2], (*iter)[3]);
                                } else {
                                    grad_tau -= m_curl_u_ref_v((*iter)[0], (*iter)[1], (*iter)[2], (*iter)[3]);
                                }

                                ++e;
                            }

                            grad_n *= m_dual_edge_length_reciprocal_v(i, c, j, k);
                            grad_tau *= m_edge_length_reciprocal_v(i, c, j, k);

                            m_lap_ref_v(i, c, j, k) = grad_n - grad_tau;
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
    };
}

#pragma once

#include "../icosahedral/unstructured_grid.hpp"
#include <random>

namespace operators {
    using namespace gridtools;
    using namespace enumtype;
    using namespace expressions;

#ifdef __CUDACC__
#define BACKEND backend< gridtools::enumtype::Cuda, gridtools::enumtype::GRIDBACKEND, gridtools::enumtype::Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< gridtools::enumtype::Host, gridtools::enumtype::GRIDBACKEND, gridtools::enumtype::Block >
#else
#define BACKEND backend< gridtools::enumtype::Host, gridtools::enumtype::GRIDBACKEND, gridtools::enumtype::Naive >
#endif
#endif

    using backend_t = BACKEND;

    class repository {
      public:
        using layout_3d_t = backend_t::select_layout< selector<1,1,1,-1> >;
        using icosahedral_topology_t = ::gridtools::icosahedral_topology< backend_t >;
        using edge_storage_type = typename backend_t::storage_t< icosahedral_topology_t::edges, double >;
        using cell_storage_type = typename backend_t::storage_t< icosahedral_topology_t::cells, double >;
        using vertex_storage_type = typename backend_t::storage_t< icosahedral_topology_t::vertexes, double >;

      private:
        icosahedral_topology_t icosahedral_grid_;

        edge_storage_type m_u;
        vertex_storage_type m_out_vertex;
        vertex_storage_type m_dual_area;
        vertex_storage_type m_curl_u_ref;

        decltype(meta_storage_extender()(m_dual_area.meta_data(), 6)) m_edges_of_vertexes_meta;

      public:
        using edges_of_vertexes_storage_type =
            typename backend_t::storage_type< double, decltype(m_edges_of_vertexes_meta) >::type;

      private:
        edge_storage_type m_edge_length;
        edge_storage_type m_dual_edge_length;
        edges_of_vertexes_storage_type m_edge_orientation;

      public:
        uint_t idim() { return m_idim; }
        uint_t jdim() { return m_jdim; }
        uint_t kdim() { return m_kdim; }
        //        typedef arg<0, edge_storage_type> p_in_edges;
        //        typedef arg<1, vertex_storage_type> p_dual_area;
        //        typedef arg<2, edge_storage_type> p_dual_edge_length;
        //        typedef arg<3, vertex_storage_type> p_out_vertexes;

        repository(const uint_t idim, const uint_t jdim, const uint_t kdim)
            : icosahedral_grid_(idim, jdim, kdim), m_idim(idim), m_jdim(jdim), m_kdim(kdim),
              m_u(icosahedral_grid_.template make_storage< icosahedral_topology_t::edges, float_type >("u")),
              m_out_vertex(icosahedral_grid_.template make_storage< icosahedral_topology_t::vertexes, float_type >(
                  "out_vertex")),
              m_dual_area(
                  icosahedral_grid_.template make_storage< icosahedral_topology_t::vertexes, float_type >("dual_area")),
              m_curl_u_ref(icosahedral_grid_.template make_storage< icosahedral_topology_t::vertexes, float_type >(
                  "curl_u_ref")),
              m_edge_length(
                  icosahedral_grid_.template make_storage< icosahedral_topology_t::edges, float_type >("edge_length")),
              m_dual_edge_length(icosahedral_grid_.template make_storage< icosahedral_topology_t::edges, float_type >(
                  "dual_edge_length")),
              m_edges_of_vertexes_meta(meta_storage_extender()(m_dual_area.meta_data(), 6)),
              m_edge_orientation(m_edges_of_vertexes_meta, "edge_orientation") {}

        void init_fields() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0, 1);
            // init u
            for (int i = 0; i < m_idim; ++i) {
                for (int c = 0; c < icosahedral_topology_t::edges::n_colors::value; ++c) {
                    for (int j = 0; j < m_jdim; ++j) {
                        for (uint_t k = 0; k < m_kdim; ++k) {
                            m_u(i, c, j, k) = 1000 * (1.0 + dis(gen));
                            m_edge_length(i, c, j, k) = (2.95 + dis(gen));
                            m_dual_edge_length(i, c, j, k) = (2.2 + dis(gen));
                        }
                    }
                }
            }

            for (int i = 0; i < m_idim; ++i) {
                for (int c = 0; c < icosahedral_topology_t::vertexes::n_colors::value; ++c) {
                    for (int j = 0; j < m_jdim; ++j) {
                        for (uint_t k = 0; k < m_kdim; ++k) {
                            m_dual_area(i, c, j, k) = 1.1 + dis(gen);
                        }
                    }
                }
            }

            // edge orientation
            for (int i = 0; i < m_idim; ++i) {
                for (int c = 0; c < icosahedral_topology_t::vertexes::n_colors::value; ++c) {
                    for (int j = 0; j < m_jdim; ++j) {
                        for (uint_t k = 0; k < m_kdim; ++k) {
                            m_edge_orientation(i, c, j, k, 0) = -1;
                            m_edge_orientation(i, c, j, k, 2) = -1;
                            m_edge_orientation(i, c, j, k, 4) = -1;
                            m_edge_orientation(i, c, j, k, 1) = 1;
                            m_edge_orientation(i, c, j, k, 3) = 1;
                            m_edge_orientation(i, c, j, k, 5) = 1;
                        }
                    }
                }
            }

            //            div_u_ref_.initialize(0.0);
            m_curl_u_ref.initialize(0.0);
            //            grad_div_u_ref_.initialize(0.0);
            //            grad_curl_u_ref_.initialize(0.0);
            //            lap_u_ref_.initialize(0.0);
        }

        inline
        void generate_reference() {
            //            // div_u_ref_
            unstructured_grid ugrid(m_idim, m_jdim, m_kdim);
            //            for (uint_t i = halo_nc; i < icosahedral_grid_.m_dims[0] - halo_nc; ++i)
            //                for (uint_t c = 0; c < icosahedral_topology_t::cells::n_colors::value; ++c)
            //                    for (uint_t j = halo_mc; j < icosahedral_grid_.m_dims[1] - halo_mc; ++j)
            //                        for (uint_t k = 0; k < d3; ++k) {
            //                            auto neighbours =
            //                                ugrid.neighbours_of< icosahedral_topology_t::cells,
            //                                icosahedral_topology_t::edges >(
            //                                    {i, c, j, k});
            //                            ushort_t e = 0;
            //                            for (auto iter = neighbours.begin(); iter != neighbours.end(); ++iter) {
            //                                div_u_ref_(i, c, j, k) +=
            //                                    orientation_of_normal_(i, c, j, k, e) * u_(*iter) *
            //                                    edge_length_(*iter);
            //                                ++e;
            //                            }
            //                            div_u_ref_(i, c, j, k) /= cell_area_(i, c, j, k);
            //                        }

            // curl_u_ref_
            for (uint_t i = halo_nc; i < m_idim - halo_nc; ++i) {
                for (uint_t c = 0; c < icosahedral_topology_t::vertexes::n_colors::value; ++c) {
                    for (uint_t j = halo_mc; j < m_jdim - halo_mc; ++j) {
                        for (uint_t k = 0; k < m_kdim; ++k) {
                            auto neighbours =
                                ugrid.neighbours_of< icosahedral_topology_t::vertexes, icosahedral_topology_t::edges >(
                                    {i, c, j, k});
                            ushort_t e = 0;
                            for (auto iter = neighbours.begin(); iter != neighbours.end(); ++iter) {
                                m_curl_u_ref(i, c, j, k) +=
                                    m_edge_orientation(i, c, j, k, e) * m_u(*iter) * m_dual_edge_length(*iter);
                                ++e;
                            }
                            m_curl_u_ref(i, c, j, k) /= m_dual_area(i, c, j, k);
                        }
                    }
                }
            }

            //            // grad_div_u_ref_
            //            for (uint_t i = halo_nc; i < icosahedral_grid_.m_dims[0] - halo_nc; ++i)
            //                for (uint_t c = 0; c < icosahedral_topology_t::cells::n_colors::value; ++c)
            //                    for (uint_t j = halo_mc; j < icosahedral_grid_.m_dims[1] - halo_mc; ++j)
            //                        for (uint_t k = 0; k < d3; ++k) {
            //                            auto neighbours_cell =
            //                                ugrid.neighbours_of< icosahedral_topology_t::cells,
            //                                icosahedral_topology_t::cells >(
            //                                    {i, c, j, k});
            //                            auto neighbours_edge =
            //                                ugrid.neighbours_of< icosahedral_topology_t::cells,
            //                                icosahedral_topology_t::edges >(
            //                                    {i, c, j, k});

            //                            ushort_t e = 0;
            //                            auto iter_edge = neighbours_edge.begin();
            //                            for (auto iter_cell = neighbours_cell.begin(); iter_cell !=
            //                            neighbours_cell.end();
            //                                 ++iter_cell, ++iter_edge) {
            //                                grad_div_u_ref_(*iter_edge) = orientation_of_normal_(i, c, j, k, e) *
            //                                                              (div_u_ref_(*iter_cell) - div_u_ref_(i, c,
            //                                                              j, k)) /
            //                                                              dual_edge_length_(*iter_edge);
            //                                ++e;
            //                            }
            //                        }

            //            // grad_curl_u_ref_
            //            for (uint_t i = halo_nc; i < icosahedral_grid_.m_dims[0] - halo_nc; ++i)
            //                for (uint_t c = 0; c < icosahedral_topology_t::vertexes::n_colors::value; ++c)
            //                    for (uint_t j = halo_mc; j < icosahedral_grid_.m_dims[1] - halo_mc; ++j)
            //                        for (uint_t k = 0; k < d3; ++k) {
            //                            auto neighbours_vertex = ugrid.neighbours_of<
            //                            icosahedral_topology_t::vertexes,
            //                                icosahedral_topology_t::vertexes >({i, c, j, k});
            //                            auto neighbours_edge =
            //                                ugrid.neighbours_of< icosahedral_topology_t::vertexes,
            //                                icosahedral_topology_t::edges >(
            //                                    {i, c, j, k});

            //                            ushort_t e = 0;
            //                            auto iter_edge = neighbours_edge.begin();
            //                            for (auto iter_vertex = neighbours_vertex.begin(); iter_vertex !=
            //                            neighbours_vertex.end();
            //                                 ++iter_vertex, ++iter_edge) {
            //                                grad_curl_u_ref_(*iter_edge) = edge_orientation_(i, c, j, k, e) *
            //                                                               (curl_u_ref_(*iter_vertex) - curl_u_ref_(i,
            //                                                               c, j, k)) /
            //                                                               edge_length_(*iter_edge);
            //                                ++e;
            //                            }
            //                        }

            //            // lap_u_ref_
            //            for (uint_t i = halo_nc; i < icosahedral_grid_.m_dims[0] - halo_nc; ++i)
            //                for (uint_t c = 0; c < icosahedral_topology_t::edges::n_colors::value; ++c)
            //                    for (uint_t j = halo_mc; j < icosahedral_grid_.m_dims[1] - halo_mc; ++j)
            //                        for (uint_t k = 0; k < d3; ++k) {
            //                            lap_u_ref_(i, c, j, k) = grad_div_u_ref_(i, c, j, k) - grad_curl_u_ref_(i, c,
            //                            j, k);
            //                        }
        }

        icosahedral_topology_t &icosahedral_grid() { return icosahedral_grid_; }

        edge_storage_type &u() { return m_u; }
        vertex_storage_type &out_vertex() { return m_out_vertex; }
        vertex_storage_type &dual_area() { return m_dual_area; }
        vertex_storage_type &curl_u_ref() { m_curl_u_ref; }

        edge_storage_type &edge_length() { return m_edge_length; }
        edge_storage_type &dual_edge_length() { return m_dual_edge_length; }

        //        edge_storage_type &edge_length() { return edge_length_; }
        //        cell_storage_type &cell_area() { return cell_area_; }
        //        edge_storage_type &u() { return u_; }
        //        cell_storage_type &div_u_ref() { return div_u_ref_; }
        //        vertex_storage_type &dual_area() { return dual_area_; }
        //        edge_storage_type &dual_edge_length() { return dual_edge_length_; }
        //        vertex_storage_type &curl_u_ref() { return curl_u_ref_; }
        decltype(m_edges_of_vertexes_meta) &edges_of_vertexes_meta() { return m_edges_of_vertexes_meta; }
        edges_of_vertexes_storage_type &edge_orientation() { return m_edge_orientation; }
        //        edges_of_cells_storage_type &orientation_of_normal() { return orientation_of_normal_; }
        //        decltype(edges_of_cells_meta_) &edges_of_cells_meta() { return edges_of_cells_meta_; }
        //        edge_storage_type &grad_div_u_ref() { return grad_div_u_ref_; }
        //        edge_storage_type &grad_curl_u_ref() { return grad_curl_u_ref_; }
        //        edge_storage_type &lap_u_ref() { return lap_u_ref_; }

      public:
        const uint_t m_idim, m_jdim, m_kdim;
        const uint_t halo_nc = 1;
        const uint_t halo_mc = 1;
        const uint_t halo_k = 0;
    };
}

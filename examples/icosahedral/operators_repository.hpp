#pragma once

#include "../icosahedral/unstructured_grid.hpp"
#include <random>

namespace ico_operators {
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
        using layout_2d_t = backend_t::select_layout< selector< 1, 1, 1, -1 > >;

        using icosahedral_topology_t = ::gridtools::icosahedral_topology< backend_t >;

        using edge_metastorage_2d_t = icosahedral_topology_t::meta_storage_2d_t< icosahedral_topology_t::edges >;
        using vertex_metastorage_2d_t = icosahedral_topology_t::meta_storage_2d_t< icosahedral_topology_t::vertexes >;

        using edge_storage_type = typename icosahedral_topology_t::storage_t< icosahedral_topology_t::edges, double >;
        using cell_storage_type = typename icosahedral_topology_t::storage_t< icosahedral_topology_t::cells, double >;
        using vertex_storage_type =
            typename icosahedral_topology_t::storage_t< icosahedral_topology_t::vertexes, double >;

        using edge_2d_storage_type = typename backend_t::storage_t< double, edge_metastorage_2d_t >;
        using vertex_2d_storage_type = typename backend_t::storage_t< double, vertex_metastorage_2d_t >;

      private:
        icosahedral_topology_t icosahedral_grid_;

        edge_storage_type m_u;
        vertex_storage_type m_out_vertex;
        vertex_storage_type m_curl_u_ref;
        vertex_2d_storage_type m_dual_area;
        vertex_2d_storage_type m_dual_area_reciprocal;

        decltype(meta_storage_extender()(m_dual_area.meta_data(), 6)) m_edges_of_vertexes_meta;

      public:
        using edges_of_vertexes_storage_type =
            typename backend_t::storage_type< double, decltype(m_edges_of_vertexes_meta) >::type;

      private:
        edge_2d_storage_type m_edge_length;
        edge_2d_storage_type m_dual_edge_length;
        edges_of_vertexes_storage_type m_edge_orientation;

      public:
        uint_t idim() { return m_idim; }
        uint_t jdim() { return m_jdim; }
        uint_t kdim() { return m_kdim; }

        repository(const uint_t idim, const uint_t jdim, const uint_t kdim)
            : icosahedral_grid_(idim, jdim, kdim), m_idim(idim), m_jdim(jdim), m_kdim(kdim),
              m_u(icosahedral_grid_.template make_storage< icosahedral_topology_t::edges, double >("u")),
              m_out_vertex(
                  icosahedral_grid_.template make_storage< icosahedral_topology_t::vertexes, double >("out_vertex")),
              m_curl_u_ref(
                  icosahedral_grid_.template make_storage< icosahedral_topology_t::vertexes, double >("curl_u_ref")),
              m_dual_area(
                  icosahedral_grid_.template make_2d_storage< icosahedral_topology_t::vertexes, double >("dual_area")),
              m_edge_length(
                  icosahedral_grid_.template make_2d_storage< icosahedral_topology_t::edges, double >("edge_length")),
              m_dual_edge_length(icosahedral_grid_.template make_2d_storage< icosahedral_topology_t::edges, double >(
                  "dual_edge_length")),
              m_edges_of_vertexes_meta(meta_storage_extender()(m_dual_area.meta_data(), 6)),
              m_edge_orientation(m_edges_of_vertexes_meta, "edge_orientation"),
              m_dual_area_reciprocal(icosahedral_grid_.make_2d_storage< icosahedral_topology_t::vertexes, double >(
                  "dual_area_reciprocal")) {}

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
                            m_edge_length(i, c, j, 0) = (2.95 + dis(gen));
                            m_dual_edge_length(i, c, j, 0) = (2.2 + dis(gen));
                        }
                    }
                }
            }

            for (int i = 0; i < m_idim; ++i) {
                for (int c = 0; c < icosahedral_topology_t::vertexes::n_colors::value; ++c) {
                    for (int j = 0; j < m_jdim; ++j) {
                        m_dual_area(i, c, j, 0) = 1.1 + dis(gen);
                    }
                }
            }

            // dual_area_reciprocal_
            for (int i = 0; i < icosahedral_grid_.m_dims[0]; ++i) {
                for (int c = 0; c < icosahedral_topology_t::vertexes::n_colors::value; ++c) {
                    for (int j = 0; j < icosahedral_grid_.m_dims[1]; ++j) {
                        m_dual_area_reciprocal(i, c, j, 0) = 1. / m_dual_area(i, c, j, 0);
                    }
                }
            }

            // edge orientation
            for (int i = 0; i < m_idim; ++i) {
                for (int c = 0; c < icosahedral_topology_t::vertexes::n_colors::value; ++c) {
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
        }

        inline void generate_reference() {
            unstructured_grid ugrid(m_idim, m_jdim, m_kdim);

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
                                    m_edge_orientation(i, c, j, 0, e) * m_u(*iter) * m_dual_edge_length(*iter);
                                ++e;
                            }
                            m_curl_u_ref(i, c, j, k) /= m_dual_area(i, c, j, 0);
                        }
                    }
                }
            }
        }

        icosahedral_topology_t &icosahedral_grid() { return icosahedral_grid_; }

        edge_storage_type &u() { return m_u; }
        vertex_storage_type &out_vertex() { return m_out_vertex; }
        vertex_storage_type &curl_u_ref() { return m_curl_u_ref; }
        vertex_2d_storage_type &dual_area() { return m_dual_area; }
        vertex_2d_storage_type &dual_area_reciprocal() { return m_dual_area_reciprocal; }

        edge_2d_storage_type &edge_length() { return m_edge_length; }
        edge_2d_storage_type &dual_edge_length() { return m_dual_edge_length; }

        decltype(m_edges_of_vertexes_meta) &edges_of_vertexes_meta() { return m_edges_of_vertexes_meta; }
        edges_of_vertexes_storage_type &edge_orientation() { return m_edge_orientation; }

      public:
        const uint_t m_idim, m_jdim, m_kdim;
        const uint_t halo_nc = 1;
        const uint_t halo_mc = 1;
        const uint_t halo_k = 0;
    };
}

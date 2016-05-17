//
// Created by Xiaolin Guo on 20.04.16.
//

#pragma once

#include "operator_examples_def.hpp"
#include "IconToGridTools.hpp"
#include "../icosahedral/unstructured_grid.hpp"
#include <random>

namespace operator_examples
{

class repository
{
public:
    using edge_storage_type = typename backend_t::storage_t<icosahedral_topology_t::edges, double>;
    using cell_storage_type = typename backend_t::storage_t<icosahedral_topology_t::cells, double>;
    using vertex_storage_type = typename backend_t::storage_t<icosahedral_topology_t::vertexes, double>;
private:
    IconToGridTools<icosahedral_topology_t> i2g_;
    icosahedral_topology_t& icosahedral_grid_;

    vertex_storage_type dual_area_;
    vertex_storage_type curl_u_ref_;

    edge_storage_type edge_length_;
    edge_storage_type u_;
    edge_storage_type dual_edge_length_;
    edge_storage_type grad_div_u_ref_;
    edge_storage_type grad_curl_u_ref_;
    edge_storage_type lap_u_ref_;

    cell_storage_type cell_area_;
    cell_storage_type div_u_ref_;

    decltype(meta_storage_extender()(dual_area_.meta_data(), 6)) edges_of_vertexes_meta_;
    decltype(meta_storage_extender()(cell_area_.meta_data(), 3)) edges_of_cells_meta_;
public:
    using edges_of_vertexes_storage_type = typename backend_t::storage_type< double, decltype(edges_of_vertexes_meta_) >::type;
    using edges_of_cells_storage_type = typename backend_t::storage_type< double, decltype(edges_of_cells_meta_) >::type;
private:
    edges_of_vertexes_storage_type edge_orientation_;
    edges_of_cells_storage_type orientation_of_normal_;

public:
    repository(char *mesh_file)
        : i2g_(mesh_file),
          icosahedral_grid_(i2g_.icosahedral_grid()),
          d3(i2g_.d3()),
          edge_length_(i2g_.get<icosahedral_topology_t::edges, double>("edge_length")),
          cell_area_(i2g_.get<icosahedral_topology_t::cells, double>("cell_area")),
          u_(icosahedral_grid_.make_storage<icosahedral_topology_t::edges, double>("u")),
          div_u_ref_(icosahedral_grid_.make_storage<icosahedral_topology_t::cells, double>("div_u_ref")),
          curl_u_ref_(icosahedral_grid_.make_storage<icosahedral_topology_t::vertexes, double>("curl_u_ref")),
          dual_area_(i2g_.get<icosahedral_topology_t::vertexes, double>("dual_area")),
          dual_edge_length_(i2g_.get<icosahedral_topology_t::edges, double>("dual_edge_length")),
          edges_of_vertexes_meta_(meta_storage_extender()(dual_area_.meta_data(), 6)),
          edges_of_cells_meta_(meta_storage_extender()(cell_area_.meta_data(), 3)),
          edge_orientation_(i2g_.get<edges_of_vertexes_storage_type, icosahedral_topology_t::vertexes>(edges_of_vertexes_meta_, "edge_orientation")),
          orientation_of_normal_(i2g_.get<edges_of_cells_storage_type, icosahedral_topology_t::cells>(edges_of_cells_meta_, "orientation_of_normal")),
          grad_div_u_ref_(icosahedral_grid_.make_storage<icosahedral_topology_t::edges, double>("grad_div_u_ref")),
          grad_curl_u_ref_(icosahedral_grid_.make_storage<icosahedral_topology_t::edges, double>("grad_curl_u_ref")),
          lap_u_ref_(icosahedral_grid_.make_storage<icosahedral_topology_t::edges, double>("lap_u_ref"))
    { }

    void init_fields()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 1);
        // init u
        for (int i = 0; i < icosahedral_grid_.m_dims[0]; ++i) {
            for (int c = 0; c < icosahedral_topology_t::edges::n_colors::value; ++c) {
                for (int j = 0; j < icosahedral_grid_.m_dims[1]; ++j) {
                    for (uint_t k = 0; k < d3; ++k)
                        u_(i, c, j, k) = 1000 * (1.0 + dis(gen));
                }
            }
        }

        div_u_ref_.initialize(0.0);
        curl_u_ref_.initialize(0.0);
        grad_div_u_ref_.initialize(0.0);
        grad_curl_u_ref_.initialize(0.0);
        lap_u_ref_.initialize(0.0);
    }

    void generate_reference()
    {
        // div_u_ref_
        unstructured_grid ugrid(icosahedral_grid_.m_dims[0], icosahedral_grid_.m_dims[1], d3);
        for (uint_t i = halo_nc; i < icosahedral_grid_.m_dims[0] - halo_nc; ++i)
            for (uint_t c = 0; c < icosahedral_topology_t::cells::n_colors::value; ++c)
                for (uint_t j = halo_mc; j < icosahedral_grid_.m_dims[1] - halo_mc; ++j)
                    for (uint_t k = 0; k < d3; ++k)
                    {
                        auto neighbours =
                            ugrid.neighbours_of<icosahedral_topology_t::cells, icosahedral_topology_t::edges>(
                                {i, c, j, k});
                        ushort_t e=0;
                        for (auto iter = neighbours.begin(); iter != neighbours.end(); ++iter)
                        {
                            div_u_ref_(i, c, j, k) += orientation_of_normal_(i, c, j, k, e) * u_(*iter) * edge_length_(*iter);
                            ++e;
                        }
                        div_u_ref_(i, c, j, k) /= cell_area_(i, c, j, k);
                    }

        // curl_u_ref_
        for (uint_t i = halo_nc; i < icosahedral_grid_.m_dims[0] - halo_nc; ++i)
            for (uint_t c = 0; c < icosahedral_topology_t::vertexes::n_colors::value; ++c)
                for (uint_t j = halo_mc; j < icosahedral_grid_.m_dims[1] - halo_mc; ++j)
                    for (uint_t k = 0; k < d3; ++k) {
                        auto neighbours =
                                ugrid.neighbours_of<icosahedral_topology_t::vertexes, icosahedral_topology_t::edges>(
                                        {i, c, j, k});
                        ushort_t e=0;
                        for (auto iter = neighbours.begin(); iter != neighbours.end(); ++iter)
                        {
                            curl_u_ref_(i, c, j, k) += edge_orientation_(i, c, j, k, e) * u_(*iter) * dual_edge_length_(*iter);
                            ++e;
                        }
                        curl_u_ref_(i, c, j, k) /= dual_area_(i, c, j, k);
                    }

        // grad_div_u_ref_
        for (uint_t i = halo_nc; i < icosahedral_grid_.m_dims[0] - halo_nc; ++i)
            for (uint_t c = 0; c < icosahedral_topology_t::cells::n_colors::value; ++c)
                for (uint_t j = halo_mc; j < icosahedral_grid_.m_dims[1] - halo_mc; ++j)
                    for (uint_t k = 0; k < d3; ++k) {
                        auto neighbours_cell =
                                ugrid.neighbours_of<icosahedral_topology_t::cells, icosahedral_topology_t::cells>(
                                        {i, c, j, k});
                        auto neighbours_edge =
                                ugrid.neighbours_of<icosahedral_topology_t::cells, icosahedral_topology_t::edges>(
                                        {i, c, j, k});

                        ushort_t e=0;
                        auto iter_edge = neighbours_edge.begin();
                        for (auto iter_cell = neighbours_cell.begin(); iter_cell != neighbours_cell.end(); ++iter_cell, ++iter_edge)
                        {
                            grad_div_u_ref_(*iter_edge) =
                                    orientation_of_normal_(i, c, j, k, e) *
                                    (div_u_ref_(*iter_cell) - div_u_ref_(i, c, j, k)) /
                                    dual_edge_length_(*iter_edge);
                            ++e;
                        }
                    }

        // grad_curl_u_ref_
        for (uint_t i = halo_nc; i < icosahedral_grid_.m_dims[0] - halo_nc; ++i)
            for (uint_t c = 0; c < icosahedral_topology_t::vertexes::n_colors::value; ++c)
                for (uint_t j = halo_mc; j < icosahedral_grid_.m_dims[1] - halo_mc; ++j)
                    for (uint_t k = 0; k < d3; ++k) {
                        auto neighbours_vertex =
                                ugrid.neighbours_of<icosahedral_topology_t::vertexes, icosahedral_topology_t::vertexes>(
                                        {i, c, j, k});
                        auto neighbours_edge =
                                ugrid.neighbours_of<icosahedral_topology_t::vertexes, icosahedral_topology_t::edges>(
                                        {i, c, j, k});

                        ushort_t e=0;
                        auto iter_edge = neighbours_edge.begin();
                        for (auto iter_vertex = neighbours_vertex.begin(); iter_vertex != neighbours_vertex.end(); ++iter_vertex, ++iter_edge)
                        {
                            grad_curl_u_ref_(*iter_edge) =
                                    edge_orientation_(i, c, j, k, e) *
                                    (curl_u_ref_(*iter_vertex) - curl_u_ref_(i, c, j, k)) /
                                    edge_length_(*iter_edge);
                            ++e;
                        }
                    }

        // lap_u_ref_
        for (uint_t i = halo_nc; i < icosahedral_grid_.m_dims[0] - halo_nc; ++i)
            for (uint_t c = 0; c < icosahedral_topology_t::edges::n_colors::value; ++c)
                for (uint_t j = halo_mc; j < icosahedral_grid_.m_dims[1] - halo_mc; ++j)
                    for (uint_t k = 0; k < d3; ++k) {
                        lap_u_ref_(i, c, j, k) = grad_div_u_ref_(i, c, j, k) - grad_curl_u_ref_(i, c, j, k);
                    }
    }

    icosahedral_topology_t& icosahedral_grid()
    {return icosahedral_grid_;}

    edge_storage_type &edge_length()
    { return edge_length_; }
    cell_storage_type &cell_area()
    { return cell_area_; }
    edge_storage_type &u()
    { return u_; }
    cell_storage_type &div_u_ref()
    { return div_u_ref_; }
    vertex_storage_type &dual_area()
    { return dual_area_; }
    edge_storage_type &dual_edge_length()
    { return dual_edge_length_; }
    vertex_storage_type &curl_u_ref()
    { return curl_u_ref_; }
    decltype(edges_of_vertexes_meta_) &edges_of_vertexes_meta()
    { return edges_of_vertexes_meta_; }
    edges_of_vertexes_storage_type &edge_orientation()
    { return edge_orientation_; }
    edges_of_cells_storage_type &orientation_of_normal()
    { return orientation_of_normal_;}
    decltype(edges_of_cells_meta_) &edges_of_cells_meta()
    { return edges_of_cells_meta_; }
    edge_storage_type &grad_div_u_ref()
    { return grad_div_u_ref_; }
    edge_storage_type &grad_curl_u_ref()
    { return grad_curl_u_ref_; }
    edge_storage_type &lap_u_ref()
    { return lap_u_ref_; }

public:
    const uint_t halo_nc = 1;
    const uint_t halo_mc = 1;
    const uint_t halo_k = 0;
    const int d3;

};
}


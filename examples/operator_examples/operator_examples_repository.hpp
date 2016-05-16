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

    repository(char *mesh_file)
        : i2g_(mesh_file),
          icosahedral_grid_(i2g_.icosahedral_grid()),
          d3(i2g_.d3),
          edge_length_(i2g_.get<icosahedral_topology_t::edges, double>("edge_length")),
          cell_area_(i2g_.get<icosahedral_topology_t::cells, double>("cell_area")),
          edge_sign_on_cell_(i2g_.icosahedral_grid().make_storage<icosahedral_topology_t::cells, double>("edge_sign_on_cell")),
          u_(i2g_.icosahedral_grid().make_storage<icosahedral_topology_t::edges, double>("u")),
          div_u_ref_(i2g_.icosahedral_grid().make_storage<icosahedral_topology_t::cells, double>("div_u_ref")),
          curl_u_ref_(i2g_.icosahedral_grid().make_storage<icosahedral_topology_t::vertexes, double>("curl_u_ref")),
          dual_area_(i2g_.get<icosahedral_topology_t::vertexes, double>("dual_area")),
          dual_edge_length_(i2g_.get<icosahedral_topology_t::edges, double>("dual_edge_length"))
    { }

    void init_fields()
    {
        // init edge_sign_on_cell
        for (int i = 0; i < icosahedral_grid_.m_dims[0]; ++i)
            for (int j = 0; j < icosahedral_grid_.m_dims[1]; ++j)
                for (uint_t k = 0; k < d3; ++k)
                {
                edge_sign_on_cell_(i, 0, j, k) = 1.;
                edge_sign_on_cell_(i, 1, j, k) = -1.;
            }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 1);
        // init u
        for (int i = 0; i < icosahedral_grid_.m_dims[0]; ++i) {
            for (int c = 0; c < icosahedral_topology_t::edges::n_colors::value; ++c) {
                for (int j = 0; j < icosahedral_grid_.m_dims[1]; ++j) {
                    for (uint_t k = 0; k < d3; ++k)
                        u_(i, c, j, k) = 1.0 + dis(gen);
                }
            }
        }

        // init div_u_ref
        div_u_ref_.initialize(0.0);

        // init curl_u_ref
        curl_u_ref_.initialize(0.0);
    }

    void generate_reference()
    {
        unstructured_grid ugrid(icosahedral_grid_.m_dims[0], icosahedral_grid_.m_dims[1], d3);
        for (uint_t i = halo_nc; i < icosahedral_grid_.m_dims[0] - halo_nc; ++i)
            for (uint_t c = 0; c < icosahedral_topology_t::cells::n_colors::value; ++c)
                for (uint_t j = halo_mc; j < icosahedral_grid_.m_dims[1] - halo_mc; ++j)
                    for (uint_t k = 0; k < d3; ++k)
                    {
                        auto neighbours =
                            ugrid.neighbours_of<icosahedral_topology_t::cells, icosahedral_topology_t::edges>(
                                {i, c, j, k});
                        for (auto iter = neighbours.begin(); iter != neighbours.end(); ++iter)
                        {
                            div_u_ref_(i, c, j, k) += u_(*iter) * edge_length_(*iter);
                        }
                        div_u_ref_(i, c, j, k) *= edge_sign_on_cell_(i, c, j, k) / cell_area_(i, c, j, k);
                    }

        for (uint_t i = halo_nc; i < icosahedral_grid_.m_dims[0] - halo_nc; ++i)
            for (uint_t c = 0; c < icosahedral_topology_t::vertexes::n_colors::value; ++c)
                for (uint_t j = halo_mc; j < icosahedral_grid_.m_dims[1] - halo_mc; ++j)
                    for (uint_t k = 0; k < d3; ++k)
                    {
                        auto neighbours =
                                ugrid.neighbours_of<icosahedral_topology_t::vertexes, icosahedral_topology_t::edges>(
                                        {i, c, j, k});
                        for (auto iter = neighbours.begin(); iter != neighbours.end(); ++iter)
                        {
                            curl_u_ref_(i, c, j, k) += u_(*iter) * dual_edge_length_(*iter);
                        }
                        curl_u_ref_(i, c, j, k) /= dual_area_(i, c, j, k);
                    }
    }

    icosahedral_topology_t& icosahedral_grid()
    {return icosahedral_grid_;}

    edge_storage_type &edge_length()
    { return edge_length_; }
    cell_storage_type &cell_area()
    { return cell_area_; }
    cell_storage_type &edge_sign_on_cell()
    { return edge_sign_on_cell_; }
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
private:
    IconToGridTools<icosahedral_topology_t> i2g_;
    icosahedral_topology_t& icosahedral_grid_;

    vertex_storage_type dual_area_;
    vertex_storage_type curl_u_ref_;

    edge_storage_type edge_length_;
    edge_storage_type u_;
    edge_storage_type dual_edge_length_;

    cell_storage_type edge_sign_on_cell_;
    cell_storage_type cell_area_;
    cell_storage_type div_u_ref_;
public:
    const uint_t halo_nc = 1;
    const uint_t halo_mc = 1;
    const uint_t halo_k = 0;
    const int d3;

};
}


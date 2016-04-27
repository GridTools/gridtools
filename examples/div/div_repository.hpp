//
// Created by Xiaolin Guo on 20.04.16.
//

#pragma once

#include "div_defs.hpp"
#include "IconToGridTools.hpp"
#include "../icosahedral/unstructured_grid.hpp"

namespace divergence
{

class repository
{
public:

    using edge_storage_type = typename backend_t::storage_t<icosahedral_topology_t::edges, double>;
    using cell_storage_type = typename backend_t::storage_t<icosahedral_topology_t::cells, double>;

    repository(char *mesh_file)
        : i2g_(mesh_file),
          icosahedral_grid_(i2g_.icosahedral_grid()),
          edge_length_(i2g_.get<icosahedral_topology_t::edges, double>("edge_length")),
          cell_area_(i2g_.get<icosahedral_topology_t::cells, double>("cell_area")),
          edge_sign_on_cell_(i2g_.icosahedral_grid().make_storage<icosahedral_topology_t::cells, double>("edge_sign_on_cell")),
          u_(i2g_.icosahedral_grid().make_storage<icosahedral_topology_t::edges, double>("u")),
          div_u_ref_(i2g_.icosahedral_grid().make_storage<icosahedral_topology_t::cells, double>("div_u_ref"))
    { }

    void init_fields()
    {
        // init edge_sign_on_cell
        for (int i = 0; i < icosahedral_grid_.m_dims[0]; ++i)
            for (int j = 0; j < icosahedral_grid_.m_dims[1]; ++j)
            {
                edge_sign_on_cell_(i, 0, j, 0) = -1.;
                edge_sign_on_cell_(i, 1, j, 0) = 1.;
            }

        // init u
        for (int i = 0; i < icosahedral_grid_.m_dims[0]; ++i) {
            for (int c = 0; c < icosahedral_topology_t::edges::n_colors::value; ++c) {
                for (int j = 0; j < icosahedral_grid_.m_dims[1]; ++j) {
                    u_(i, c, j, 0) = (uint_t)u_.meta_data().index(
                            array< uint_t, 4 >{(uint_t)i, (uint_t)c, (uint_t)j, (uint_t)1});
                }
            }
        }

        // init div_u_ref
        div_u_ref_.initialize(0.0);
    }

    void generate_reference()
    {
        unstructured_grid ugrid(icosahedral_grid_.m_dims[0], icosahedral_grid_.m_dims[1], 1);
        for (uint_t i = halo_nc; i < icosahedral_grid_.m_dims[0] - halo_nc; ++i)
            for (uint_t c = 0; c < icosahedral_topology_t::cells::n_colors::value; ++c)
                for (uint_t j = halo_mc; j < icosahedral_grid_.m_dims[1] - halo_mc; ++j)
                    for (uint_t k = 0; k < 1; ++k)
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

private:
    IconToGridTools<icosahedral_topology_t> i2g_;
    icosahedral_topology_t& icosahedral_grid_;
    edge_storage_type edge_length_;
    cell_storage_type edge_sign_on_cell_;
    cell_storage_type cell_area_;
    edge_storage_type u_;
    cell_storage_type div_u_ref_;

public:
    const uint_t halo_nc = 1;
    const uint_t halo_mc = 1;
    const uint_t halo_k = 0;

};
}


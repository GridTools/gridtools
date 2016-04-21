//
// Created by Xiaolin Guo on 20.04.16.
//

#pragma once

#include <gridtools.hpp>
#include "div_defs.hpp"
#include "IconToGridTools.hpp"

using gridtools::uint_t;
using gridtools::int_t;

namespace divergence
{

class repository
{
public:

    using edge_storage_type = typename backend_t::storage_t<icosahedral_topology_t::edges, double>;
    using cell_storage_type = typename backend_t::storage_t<icosahedral_topology_t::cells, double>;

    repository(char *mesh_file)
        : i2g_(mesh_file),
          edge_length_(i2g_.get<icosahedral_topology_t::edges, double>("edge_length")),
          cell_area_(i2g_.get<icosahedral_topology_t::cells, double>("cell_area")),
          u_(i2g_.icosahedral_grid().make_storage<icosahedral_topology_t::edges, double>("u")),
          div_u_ref_(i2g_.icosahedral_grid().make_storage<icosahedral_topology_t::cells, double>("div_u_ref"))
    { }

    void init_fields()
    { }

    void generate_reference()
    { }

    edge_storage_type &edge_length()
    { return edge_length_; }
    cell_storage_type &cell_area()
    { return cell_area_; }
    edge_storage_type &u()
    { return u_; }
    cell_storage_type &div_u_ref()
    { return div_u_ref_; }

private:
    IconToGridTools<icosahedral_topology_t> i2g_;
    edge_storage_type edge_length_;
    cell_storage_type cell_area_;
    edge_storage_type u_;
    cell_storage_type div_u_ref_;
};
}


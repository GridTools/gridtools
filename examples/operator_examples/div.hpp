//
// Created by Xiaolin Guo on 19.04.16.
//
#pragma once

#include "operator_examples_repository.hpp"
#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include "tools/verifier.hpp"

namespace operator_examples {

    typedef gridtools::interval< level<0, -1>, level<1, -1> > x_interval;
    typedef gridtools::interval< level<0, -2>, level<1, 1> > axis;

    struct div_functor {
        typedef in_accessor<0, icosahedral_topology_t::edges, extent<1> > in_edges;
        typedef in_accessor<1, icosahedral_topology_t::edges, extent<1> > edge_length;
        typedef inout_accessor<2, icosahedral_topology_t::cells> out_cells;
        typedef in_accessor<3, icosahedral_topology_t::cells, extent<1> > cell_area;
        typedef in_accessor<4, icosahedral_topology_t::cells, extent<1> > edge_sign_on_cell;
        typedef boost::mpl::vector<in_edges, edge_length, out_cells, cell_area, edge_sign_on_cell> arg_list;

        template<typename Evaluation>
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval)
        {
            auto ff = [](const double _in1, const double _in2, const double _res) -> double { return _in1 * _in2 + _res; };

            /**
               This interface checks that the location types are compatible with the accessors
             */
            eval(out_cells()) = eval(on_edges(ff, 0.0, in_edges(), edge_length())) * eval(edge_sign_on_cell()) / eval(cell_area());
        }
    };

    bool test_div( uint_t t_steps, char *mesh_file)
    {
        operator_examples::repository repository(mesh_file);
        repository.init_fields();
        repository.generate_reference();

        icosahedral_topology_t& icosahedral_grid = repository.icosahedral_grid();
        uint_t d1 = icosahedral_grid.m_dims[0];
        uint_t d2 = icosahedral_grid.m_dims[1];
        uint_t d3 = repository.d3;

        const uint_t halo_nc = repository.halo_nc;
        const uint_t halo_mc = repository.halo_mc;
        const uint_t halo_k = repository.halo_k;

        typedef gridtools::layout_map<2, 1, 0> layout_t;

        using edge_storage_type = repository::edge_storage_type;
        using cell_storage_type = repository::cell_storage_type;


        auto& in_edges = repository.u();
        auto& cell_area = repository.cell_area();
        auto& edge_sign = repository.edge_sign_on_cell();
        auto& edge_length = repository.edge_length();
        auto& ref_cells = repository.div_u_ref();
        auto out_cells = icosahedral_grid.make_storage<icosahedral_topology_t::cells, double>("out");

        out_cells.initialize(0.0);

        typedef arg<0, edge_storage_type> p_in_edges;
        typedef arg<1, cell_storage_type> p_out_cells;
        typedef arg<2, cell_storage_type> p_cell_area;
        typedef arg<3, cell_storage_type> p_edge_sign;
        typedef arg<4, edge_storage_type> p_edge_length;

        typedef boost::mpl::vector<p_in_edges, p_out_cells, p_cell_area, p_edge_sign, p_edge_length>
            accessor_list_t;

        gridtools::domain_type<accessor_list_t> domain(
            boost::fusion::make_vector(&in_edges, &out_cells, &cell_area, &edge_sign, &edge_length));
        array<uint_t, 5> di = {halo_nc, halo_nc, halo_nc, d1 - halo_nc - 1, d1};
        array<uint_t, 5> dj = {halo_mc, halo_mc, halo_mc, d2 - halo_mc - 1, d2};

        gridtools::grid<axis, icosahedral_topology_t> grid_(icosahedral_grid, di, dj);
        grid_.value_list[0] = 0;
        grid_.value_list[1] = d3 - 1;

        auto stencil_ = gridtools::make_computation<backend_t>(
            domain,
            grid_,
            gridtools::make_mss // mss_descriptor
                (execute<forward>(),
                 gridtools::make_esf<div_functor, icosahedral_topology_t, icosahedral_topology_t::cells>(
                         p_in_edges(), p_edge_length(), p_out_cells(), p_cell_area(), p_edge_sign())));
        stencil_->ready();
        stencil_->steady();
        stencil_->run();

    #ifdef __CUDACC__
        out_edges.d2h_update();
            in_edges.d2h_update();
    #endif

        verifier ver(1e-10);

        array<array<uint_t, 2>, 4> halos = {{{halo_nc, halo_nc}, {0, 0}, {halo_mc, halo_mc}, {halo_k, halo_k}}};
        bool result = ver.verify(grid_, ref_cells, out_cells, halos);

    #ifdef BENCHMARK
        for (uint_t t = 1; t < t_steps; ++t)
        {
            stencil_->run();
        }
        stencil_->finalize();
        std::cout << stencil_->print_meter() << std::endl;
    #endif

        return result;
    }
}

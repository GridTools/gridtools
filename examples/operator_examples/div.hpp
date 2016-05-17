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

    struct div_prep_functor {
        typedef in_accessor<0, icosahedral_topology_t::edges, extent<1> > edge_length;
        typedef in_accessor<1, icosahedral_topology_t::cells, extent<1> > cell_area;
        typedef in_accessor<2, icosahedral_topology_t::cells, extent<1>, 5 > orientation_of_normal;
        typedef inout_accessor<3, icosahedral_topology_t::cells, 5 > weights;
        typedef boost::mpl::vector<edge_length, cell_area, orientation_of_normal, weights> arg_list;

        template<typename Evaluation>
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval)
        {
            typedef typename icgrid::get_grid_topology< Evaluation >::type grid_topology_t;

            using edge_of_cell_dim = dimension< 5 >;
            edge_of_cell_dim::Index edge;

            auto neighbors_offsets = connectivity< cells , edges >::offsets(eval.position()[1]);
            ushort_t e=0;
            for (auto neighbor_offset : neighbors_offsets) {
                eval(weights(edge + e)) += eval(orientation_of_normal(edge + e)) * eval(edge_length(neighbor_offset)) / eval(cell_area());
                e++;
            }
        }

    };

    struct div_functor {
        typedef in_accessor<0, icosahedral_topology_t::edges, extent<1> > in_edges;
        typedef in_accessor<1, icosahedral_topology_t::cells, extent<1>, 5 > weights;
        typedef inout_accessor<2, icosahedral_topology_t::cells> out_cells;
        typedef boost::mpl::vector<in_edges, weights, out_cells> arg_list;

        template<typename Evaluation>
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval)
        {
            typedef typename icgrid::get_grid_topology< Evaluation >::type grid_topology_t;

            using edge_of_cells_dim = dimension< 5 >;
            edge_of_cells_dim::Index edge;

            eval(out_cells()) = 0.;
            auto neighbors_offsets = connectivity< cells , edges >::offsets(eval.position()[1]);
            ushort_t e=0;
            for (auto neighbor_offset : neighbors_offsets) {
                eval(out_cells()) += eval(in_edges(neighbor_offset)) * eval(weights(edge+e));
                e++;
            }
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
        using edges_of_cells_storage_type = repository::edges_of_cells_storage_type;


        auto& in_edges = repository.u();
        auto& cell_area = repository.cell_area();
        edges_of_cells_storage_type& orientation_of_normal = repository.orientation_of_normal();
        auto& edge_length = repository.edge_length();
        auto& ref_cells = repository.div_u_ref();
        auto out_cells = icosahedral_grid.make_storage<icosahedral_topology_t::cells, double>("out");

        auto& weights_meta = repository.edges_of_cells_meta();
        edges_of_cells_storage_type div_weights(weights_meta, "weights");

        out_cells.initialize(0.0);
        div_weights.initialize(0.0);

        typedef arg<0, edge_storage_type> p_in_edges;
        typedef arg<1, cell_storage_type> p_out_cells;
        typedef arg<2, cell_storage_type> p_cell_area;
        typedef arg<3, edges_of_cells_storage_type> p_orientation_of_normal;
        typedef arg<4, edge_storage_type> p_edge_length;
        typedef arg<5, edges_of_cells_storage_type> p_div_weights;

        typedef boost::mpl::vector<p_in_edges, p_out_cells, p_cell_area, p_orientation_of_normal, p_edge_length, p_div_weights>
            accessor_list_t;

        gridtools::domain_type<accessor_list_t> domain(
            boost::fusion::make_vector(&in_edges, &out_cells, &cell_area, &orientation_of_normal, &edge_length, &div_weights));
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
                 gridtools::make_esf<div_prep_functor, icosahedral_topology_t, icosahedral_topology_t::cells>(
                         p_edge_length(), p_cell_area(), p_orientation_of_normal(), p_div_weights()),
                 gridtools::make_esf<div_functor, icosahedral_topology_t, icosahedral_topology_t::cells>(
                         p_in_edges(), p_div_weights(), p_out_cells())
                )
        );
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

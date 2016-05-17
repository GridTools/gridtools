//
// Created by Xiaolin Guo on 17.05.16.
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

    struct grad_n {
        typedef in_accessor<0, icosahedral_topology_t::cells, extent<1> > in_cells;
        typedef in_accessor<1, icosahedral_topology_t::vertexes, extent<1>, 5 > orientation_of_normal;
        typedef inout_accessor<2, icosahedral_topology_t::edges > out_edges;
        typedef in_accessor<3, icosahedral_topology_t::edges, extent<1> > dual_edge_length;
        typedef boost::mpl::vector<in_cells, orientation_of_normal, out_edges, dual_edge_length> arg_list;

        template<typename Evaluation>
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval)
        {
            typedef typename icgrid::get_grid_topology< Evaluation >::type grid_topology_t;

            using edge_of_cell_dim = dimension< 5 >;
            edge_of_cell_dim::Index edge;

            auto neighbors_offsets = connectivity< cells, cells >::offsets(eval.position()[1]);
            auto neighbors_edge_offsets = connectivity< cells, edges >::offsets(eval.position()[1]);
            auto it_neighbors_edge_offsets = neighbors_edge_offsets.begin();
            ushort_t e=0;
            for (auto neighbor_offset : neighbors_offsets) {
                eval(out_edges(*it_neighbors_edge_offsets)) =
                        eval(orientation_of_normal(edge+e)) *
                        (eval(in_cells(neighbor_offset)) - eval(in_cells())) /
                        eval(dual_edge_length(*it_neighbors_edge_offsets));
                e++;
                ++it_neighbors_edge_offsets;
            }
        }
    };

    bool test_grad_n( uint_t t_steps, char *mesh_file)
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


        auto& in_cells = repository.div_u_ref();
        auto out_edges = icosahedral_grid.make_storage<icosahedral_topology_t::edges, double>("out");
        auto& ref_edges = repository.grad_div_u_ref();
        edges_of_cells_storage_type& orientation_of_normal = repository.orientation_of_normal();
        auto& dual_edge_length = repository.dual_edge_length();

        out_edges.initialize(0.0);

        typedef arg<0, cell_storage_type> p_in_cells;
        typedef arg<1, edge_storage_type> p_out_edges;
        typedef arg<2, edges_of_cells_storage_type> p_orientation_of_normal;
        typedef arg<3, edge_storage_type> p_dual_edge_length;

        typedef boost::mpl::vector<p_in_cells, p_out_edges, p_orientation_of_normal, p_dual_edge_length>
                accessor_list_t;

        gridtools::domain_type<accessor_list_t> domain(
                boost::fusion::make_vector(&in_cells, &out_edges, &orientation_of_normal, &dual_edge_length)
        );
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
                         gridtools::make_esf<grad_n, icosahedral_topology_t, icosahedral_topology_t::cells >(
                                 p_in_cells(), p_orientation_of_normal(), p_out_edges(), p_dual_edge_length())
                        )
        );

        stencil_->ready();
        stencil_->steady();
        stencil_->run();

#ifdef __CUDACC__
        out_edges.d2h_update();
        in_cells.d2h_update();
#endif

        verifier ver(1e-10);

        array<array<uint_t, 2>, 4> halos = {{{halo_nc, halo_nc}, {0, 0}, {halo_mc, halo_mc}, {halo_k, halo_k}}};
        bool result = ver.verify(grid_, ref_edges, out_edges, halos);

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


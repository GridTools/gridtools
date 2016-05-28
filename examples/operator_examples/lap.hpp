//
// Created by Xiaolin Guo on 17.05.16.
//

#pragma once

#include "div.hpp"
#include "curl.hpp"
#include "grad_n.hpp"
#include "grad_tau.hpp"

namespace operator_examples {

    typedef gridtools::interval<level<0, -1>, level<1, -1> >
            x_interval;
    typedef gridtools::interval<level<0, -2>, level<1, 1> >
            axis;

    struct lap_functor {
        typedef in_accessor<0, icosahedral_topology_t::edges, extent<1> > grad_div;
        typedef in_accessor<1, icosahedral_topology_t::edges, extent<1> > grad_curl;
        typedef inout_accessor<2, icosahedral_topology_t::edges> out_edges;
        typedef boost::mpl::vector<grad_div, grad_curl, out_edges> arg_list;

        template<typename Evaluation>
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(out_edges()) = eval(grad_div()) - eval(grad_curl());
        }
    };

    bool test_lap(uint_t t_steps, char *mesh_file) {
        operator_examples::repository repository(mesh_file);
        repository.init_fields();
        repository.generate_reference();

        icosahedral_topology_t &icosahedral_grid = repository.icosahedral_grid();
        uint_t d1 = icosahedral_grid.m_dims[0];
        uint_t d2 = icosahedral_grid.m_dims[1];
        uint_t d3 = repository.d3;

        const uint_t halo_nc = repository.halo_nc;
        const uint_t halo_mc = repository.halo_mc;
        const uint_t halo_k = repository.halo_k;

        typedef gridtools::layout_map<2, 1, 0> layout_t;

        using edge_storage_type = repository::edge_storage_type;
        using cell_storage_type = repository::cell_storage_type;
        using vertex_storage_type = repository::vertex_storage_type;
        using edges_of_vertexes_storage_type = repository::edges_of_vertexes_storage_type;
        using edges_of_cells_storage_type = repository::edges_of_cells_storage_type;

        // for curl
        auto &dual_area = repository.dual_area();
        auto &dual_edge_length = repository.dual_edge_length();
        auto &curl_weights_meta = repository.edges_of_vertexes_meta();
        edges_of_vertexes_storage_type curl_weights(curl_weights_meta, "curl_weights");
        edges_of_vertexes_storage_type &edge_orientation = repository.edge_orientation();

        curl_weights.initialize(0.0);

        // for div
        auto &cell_area = repository.cell_area();
        auto &edge_length = repository.edge_length();
        auto &div_weights_meta = repository.edges_of_cells_meta();
        edges_of_cells_storage_type div_weights(div_weights_meta, "div_weights");
        edges_of_cells_storage_type &orientation_of_normal = repository.orientation_of_normal();

        div_weights.initialize(0.0);

        typedef arg<0, vertex_storage_type> p_dual_area;
        typedef arg<1, edge_storage_type> p_dual_edge_length;
        typedef arg<2, edges_of_vertexes_storage_type> p_curl_weights;
        typedef arg<3, edges_of_vertexes_storage_type> p_edge_orientation;

        typedef arg<4, cell_storage_type> p_cell_area;
        typedef arg<5, edge_storage_type> p_edge_length;
        typedef arg<6, edges_of_cells_storage_type> p_div_weights;
        typedef arg<7, edges_of_cells_storage_type> p_orientation_of_normal;

        typedef boost::mpl::vector<
                p_dual_area,
                p_dual_edge_length,
                p_curl_weights,
                p_edge_orientation,
                p_cell_area,
                p_edge_length,
                p_div_weights,
                p_orientation_of_normal
        > accessor_list_prep_t;

        gridtools::domain_type<accessor_list_prep_t> prep_domain(
                boost::fusion::make_vector(
                        &dual_area,
                        &dual_edge_length,
                        &curl_weights,
                        &edge_orientation,
                        &cell_area,
                        &edge_length,
                        &div_weights,
                        &orientation_of_normal
                )
        );

        array<uint_t, 5> di = {halo_nc, halo_nc, halo_nc, d1 - halo_nc - 1, d1};
        array<uint_t, 5> dj = {halo_mc, halo_mc, halo_mc, d2 - halo_mc - 1, d2};

        gridtools::grid<axis, icosahedral_topology_t> grid_(icosahedral_grid, di, dj);
        grid_.value_list[0] = 0;
        grid_.value_list[1] = d3 - 1;

        auto prep_stencil_ = gridtools::make_computation<backend_t>(
                prep_domain,
                grid_,
                gridtools::make_mss(
                        execute<forward>(),
//                        make_independent(
                        make_esf<div_prep_functor, icosahedral_topology_t, icosahedral_topology_t::cells>(
                                p_edge_length(), p_cell_area(), p_orientation_of_normal(), p_div_weights()),
                        make_esf<curl_prep_functor, icosahedral_topology_t, icosahedral_topology_t::vertexes>(
                                p_dual_area(), p_dual_edge_length(), p_curl_weights(), p_edge_orientation())
//                        )
                )
        );

        prep_stencil_->ready();
        prep_stencil_->steady();
        prep_stencil_->run();

        /*
         * prep stage is done
         */

        auto &in_edges = repository.u();
        auto div_on_cells = icosahedral_grid.make_storage<icosahedral_topology_t::cells, double>("div_on_cells");
        auto grad_div = icosahedral_grid.make_storage<icosahedral_topology_t::edges, double>("grad_div");
        auto curl_on_vertexes = icosahedral_grid.make_storage<icosahedral_topology_t::vertexes, double>(
                "curl_on_vertexes");
        auto grad_curl = icosahedral_grid.make_storage<icosahedral_topology_t::edges, double>("grad_curl");
        auto out_edges = icosahedral_grid.make_storage<icosahedral_topology_t::edges, double>("out");
        auto &ref_edges = repository.lap_u_ref();
        div_on_cells.initialize(0.0);
        curl_on_vertexes.initialize(0.0);
        out_edges.initialize(0.0);

        typedef arg<0, edge_storage_type> p_in_edges;
        typedef arg<1, cell_storage_type> p_div_on_cells;
        typedef arg<2, edge_storage_type> p_grad_div;
        typedef arg<3, vertex_storage_type> p_curl_on_vertexes;
        typedef arg<4, edge_storage_type> p_grad_curl;
        typedef arg<5, edge_storage_type> p_out_edges;
        typedef arg<6, edges_of_vertexes_storage_type> p_curl_weights1;
        typedef arg<7, edges_of_cells_storage_type> p_div_weights1;
        typedef arg<8, edges_of_cells_storage_type> p_orientation_of_normal1;
        typedef arg<9, edge_storage_type> p_dual_edge_length1;
        typedef arg<10, edges_of_vertexes_storage_type> p_edge_orientation1;
        typedef arg<11, edge_storage_type> p_edge_length1;

        typedef boost::mpl::vector<
                p_in_edges,
                p_div_on_cells,
                p_grad_div,
                p_curl_on_vertexes,
                p_grad_curl,
                p_out_edges,
                p_curl_weights1,
                p_div_weights1,
                p_orientation_of_normal1,
                p_dual_edge_length1,
                p_edge_orientation1,
                p_edge_length1
        > accessor_list_t;

        gridtools::domain_type<accessor_list_t> domain(
                boost::fusion::make_vector(
                        &in_edges,
                        &div_on_cells,
                        &grad_div,
                        &curl_on_vertexes,
                        &grad_curl,
                        &out_edges,
                        &curl_weights,
                        &div_weights,
                        &orientation_of_normal,
                        &dual_edge_length,
                        &edge_orientation,
                        &edge_length
                )
        );

        auto stencil_ = gridtools::make_computation<backend_t>(
                domain,
                grid_,
                gridtools::make_mss(
                        execute<forward>(),
                        make_esf<div_functor, icosahedral_topology_t, icosahedral_topology_t::cells>(
                                p_in_edges(), p_div_weights1(), p_div_on_cells()
                        ),
                        make_esf<curl_functor, icosahedral_topology_t, icosahedral_topology_t::vertexes>(
                                p_in_edges(), p_curl_weights1(), p_curl_on_vertexes()
                        ),
                        make_esf<grad_n, icosahedral_topology_t, icosahedral_topology_t::cells>(
                                p_div_on_cells(), p_orientation_of_normal1(), p_grad_div(), p_dual_edge_length1()
                        ),
                        make_esf<grad_tau, icosahedral_topology_t, icosahedral_topology_t::vertexes>(
                                p_curl_on_vertexes(), p_edge_orientation1(), p_grad_curl(), p_edge_length1()
                        ),
                        make_esf<lap_functor, icosahedral_topology_t, icosahedral_topology_t::edges>(
                                p_grad_div(), p_grad_curl(), p_out_edges()
                        )
                )
        );

        stencil_->ready();
        stencil_->steady();
        stencil_->run();

#ifdef __CUDACC__
        in_edges.d2h_update();
        out_edges.d2h_update();
#endif

        verifier ver(1e-15);

        array<array<uint_t, 2>, 4> halos = {{{halo_nc, halo_nc}, {0, 0}, {halo_mc, halo_mc}, {halo_k, halo_k}}};
        bool result = ver.verify(grid_, ref_edges, out_edges, halos);

#ifdef BENCHMARK
        for (uint_t t = 1; t < t_steps; ++t) {
            stencil_->run();
        }
        stencil_->finalize();
        std::cout << stencil_->print_meter() << std::endl;
#endif

        return result;
    }
}

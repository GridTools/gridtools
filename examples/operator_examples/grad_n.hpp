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

    typedef gridtools::interval<level < 0, -1>, level<1, -1> >
    x_interval;
    typedef gridtools::interval<level < 0, -2>, level<1, 1> >
    axis;

    struct grad_n {
        typedef in_accessor<0, icosahedral_topology_t::cells, extent<1> > in_cells;
        typedef in_accessor<1, icosahedral_topology_t::cells, extent<1>, 5> orientation_of_normal;
        typedef inout_accessor<2, icosahedral_topology_t::edges> out_edges;
        typedef in_accessor<3, icosahedral_topology_t::edges, extent<1> > dual_edge_length;
        typedef boost::mpl::vector<in_cells, orientation_of_normal, out_edges, dual_edge_length> arg_list;

        template<typename Evaluation>
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            typedef typename icgrid::get_grid_topology<Evaluation>::type grid_topology_t;

            using edge_of_cell_dim = dimension<5>;
            edge_of_cell_dim::Index edge;

            auto neighbors_offsets = connectivity<cells, cells>::offsets(eval.position()[1]);
            auto neighbors_edge_offsets = connectivity<cells, edges>::offsets(eval.position()[1]);
            auto it_neighbors_edge_offsets = neighbors_edge_offsets.begin();
            ushort_t e = 0;
            for (auto neighbor_offset : neighbors_offsets) {
                eval(out_edges(*it_neighbors_edge_offsets)) =
                        eval(orientation_of_normal(edge + e)) *
                        (eval(in_cells(neighbor_offset)) - eval(in_cells())) /
                        eval(dual_edge_length(*it_neighbors_edge_offsets));
                e++;
                ++it_neighbors_edge_offsets;
            }
        }
    };

    struct grad_n_prep {
        typedef in_accessor<0, icosahedral_topology_t::cells, extent<0>, 5> orientation_of_normal;
        typedef in_accessor<1, icosahedral_topology_t::edges, extent<0> > dual_edge_length;
        typedef inout_accessor<2, icosahedral_topology_t::cells, 5> weights;
        typedef inout_accessor<3, icosahedral_topology_t::edges> dual_edge_length_reciprocal;
        typedef boost::mpl::vector<orientation_of_normal, dual_edge_length, weights, dual_edge_length_reciprocal> arg_list;

        template<typename Evaluation>
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            if (eval.position()[1] == 0) {
                constexpr auto neighbors_offsets = from<edges>::to<cells>::with_color<static_int<0> >::offsets();
                double weight =
                        (double) eval(orientation_of_normal(neighbors_offsets[0].append_dim(2))) /
                        eval(dual_edge_length());
                eval(weights(weights(neighbors_offsets[0].append_dim(2)))) = weight;
                eval(weights(weights(neighbors_offsets[1].append_dim(2)))) = -weight;
            } else if (eval.position()[1] == 1) {
                constexpr auto neighbors_offsets = from<edges>::to<cells>::with_color<static_int<1> >::offsets();
                double weight =
                        (double) eval(orientation_of_normal(neighbors_offsets[0].append_dim(0))) /
                        eval(dual_edge_length());
                eval(weights(weights(neighbors_offsets[0].append_dim(0)))) = weight;
                eval(weights(weights(neighbors_offsets[1].append_dim(0)))) = -weight;
            }
            else {
                constexpr auto neighbors_offsets = from<edges>::to<cells>::with_color<static_int<2> >::offsets();
                double weight =
                        (double) eval(orientation_of_normal(neighbors_offsets[0].append_dim(1))) /
                        eval(dual_edge_length());
                eval(weights(weights(neighbors_offsets[0].append_dim(1)))) = weight;
                eval(weights(weights(neighbors_offsets[1].append_dim(1)))) = -weight;
            }

            eval(dual_edge_length_reciprocal()) = 1.0 / eval(dual_edge_length());
        }
    };

    struct grad_n_flow_convention_reciprocal {
        typedef in_accessor<0, icosahedral_topology_t::cells, extent<1> > in_cells;
        typedef in_accessor<1, icosahedral_topology_t::edges, extent<1> > dual_edge_length_reciprocal;
        typedef inout_accessor<2, icosahedral_topology_t::edges> out_edges;
        typedef boost::mpl::vector<in_cells, dual_edge_length_reciprocal, out_edges> arg_list;

        template<typename Evaluation>
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            auto neighbors_offsets = connectivity<edges, cells>::offsets(eval.position()[1]);
            eval(out_edges()) = (eval(in_cells(neighbors_offsets[0])) - eval(in_cells(neighbors_offsets[1]))) *
                                eval(dual_edge_length_reciprocal());
        }
    };

    struct grad_n_flow_convention {
        typedef in_accessor<0, icosahedral_topology_t::cells, extent<1> > in_cells;
        typedef in_accessor<1, icosahedral_topology_t::edges, extent<1> > dual_edge_length;
        typedef inout_accessor<2, icosahedral_topology_t::edges> out_edges;
        typedef boost::mpl::vector<in_cells, dual_edge_length, out_edges> arg_list;

        template<typename Evaluation>
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            auto neighbors_offsets = connectivity<edges, cells>::offsets(eval.position()[1]);
            eval(out_edges()) = (eval(in_cells(neighbors_offsets[0])) - eval(in_cells(neighbors_offsets[1]))) /
                                eval(dual_edge_length());
        }
    };

    struct grad_n_flow_convention_with_constexpr {
        typedef in_accessor<0, icosahedral_topology_t::cells, extent<1> > in_cells;
        typedef in_accessor<1, icosahedral_topology_t::edges, extent<1> > dual_edge_length;
        typedef inout_accessor<2, icosahedral_topology_t::edges> out_edges;
        typedef boost::mpl::vector<in_cells, dual_edge_length, out_edges> arg_list;

        template<typename Evaluation>
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            if (eval.position()[1] == 0) {
                constexpr auto neighbors_offsets = from<edges>::to<cells>::with_color<static_int<0> >::offsets();
                eval(out_edges()) = (eval(in_cells(neighbors_offsets[0])) - eval(in_cells(neighbors_offsets[1]))) /
                                    eval(dual_edge_length());
            }
            else if (eval.position()[1] == 1) {
                constexpr auto neighbors_offsets = from<edges>::to<cells>::with_color<static_int<1> >::offsets();
                eval(out_edges()) = (eval(in_cells(neighbors_offsets[0])) - eval(in_cells(neighbors_offsets[1]))) /
                                    eval(dual_edge_length());
            }
            else {
                constexpr auto neighbors_offsets = from<edges>::to<cells>::with_color<static_int<2> >::offsets();
                eval(out_edges()) = (eval(in_cells(neighbors_offsets[0])) - eval(in_cells(neighbors_offsets[1]))) /
                                    eval(dual_edge_length());
            }
        }
    };

    template <int color>
    struct grad_n_flow_convention_three_esfs {
        typedef in_accessor<0, icosahedral_topology_t::cells, extent<1> > in_cells;
        typedef in_accessor<1, icosahedral_topology_t::edges, extent<1> > dual_edge_length;
        typedef inout_accessor<2, icosahedral_topology_t::edges> out_edges;
        typedef boost::mpl::vector<in_cells, dual_edge_length, out_edges> arg_list;

        template<typename Evaluation>
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            constexpr auto neighbors_offsets = from<edges>::to<cells>::with_color<static_int<color> >::offsets();
            eval(out_edges()) = (eval(in_cells(neighbors_offsets[0])) - eval(in_cells(neighbors_offsets[1]))) /
                                eval(dual_edge_length());
        }
    };

    struct grad_n_orientation_of_normal {
        typedef in_accessor<0, icosahedral_topology_t::cells, extent<1> > in_cells;
        typedef in_accessor<1, icosahedral_topology_t::cells, extent<1>, 5> weights;
        typedef inout_accessor<2, icosahedral_topology_t::edges> out_edges;
        typedef boost::mpl::vector<in_cells, weights, out_edges> arg_list;

        template<typename Evaluation>
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {

            if (eval.position()[1] == 0) {
                constexpr auto neighbors_offsets = from<edges>::to<cells>::with_color<static_int<0> >::offsets();
                eval(out_edges()) =
                        eval(weights(neighbors_offsets[0].append_dim(2))) *
                        (eval(in_cells(neighbors_offsets[1])) - eval(in_cells(neighbors_offsets[0])));
            } else if (eval.position()[1] == 1) {
                constexpr auto neighbors_offsets = from<edges>::to<cells>::with_color<static_int<1> >::offsets();
                eval(out_edges()) =
                        eval(weights(neighbors_offsets[0].append_dim(0))) *
                        (eval(in_cells(neighbors_offsets[1])) - eval(in_cells(neighbors_offsets[0])));
            }
            else {
                constexpr auto neighbors_offsets = from<edges>::to<cells>::with_color<static_int<2> >::offsets();
                eval(out_edges()) =
                        eval(weights(neighbors_offsets[0].append_dim(1))) *
                        (eval(in_cells(neighbors_offsets[1])) - eval(in_cells(neighbors_offsets[0])));
            }
        }
    };

    bool test_grad_n(uint_t t_steps, char *mesh_file) {
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
        array<array<uint_t, 2>, 4> halos = {{{halo_nc, halo_nc}, {0, 0}, {halo_mc, halo_mc}, {halo_k, halo_k}}};

        typedef gridtools::layout_map<2, 1, 0> layout_t;

        array<uint_t, 5> di = {halo_nc, halo_nc, halo_nc, d1 - halo_nc - 1, d1};
        array<uint_t, 5> dj = {halo_mc, halo_mc, halo_mc, d2 - halo_mc - 1, d2};

        verifier ver(1e-15);

        gridtools::grid<axis, icosahedral_topology_t> grid_(icosahedral_grid, di, dj);
        grid_.value_list[0] = 0;
        grid_.value_list[1] = d3 - 1;

        /*
         * initialize fields
         */

        using edge_storage_type = repository::edge_storage_type;
        using cell_storage_type = repository::cell_storage_type;
        using edges_of_cells_storage_type = repository::edges_of_cells_storage_type;

        auto &in_cells = repository.div_u_ref();
        auto &ref_edges = repository.grad_div_u_ref();
        auto &orientation_of_normal = repository.orientation_of_normal();
        auto &dual_edge_length = repository.dual_edge_length();

        auto &weights_meta = repository.edges_of_cells_meta();
        edges_of_cells_storage_type weights(weights_meta, "weights");

        auto dual_edge_length_reciprocal = icosahedral_grid.make_storage<icosahedral_topology_t::edges, double>(
                "dual_edge_length_reciprocal");
        auto out_edges = icosahedral_grid.make_storage<icosahedral_topology_t::edges, double>("out");

        dual_edge_length_reciprocal.initialize(0.0);
        out_edges.initialize(0.0);

        /*
         *  prep stencil
         */

        typedef arg<0, edges_of_cells_storage_type> p_orientation_of_normal;
        typedef arg<1, edge_storage_type> p_dual_edge_length;
        typedef arg<2, edges_of_cells_storage_type> p_weights;
        typedef arg<3, edge_storage_type> p_dual_edge_length_reciprocal;
        typedef arg<4, cell_storage_type> p_in_cells;
        typedef arg<5, edge_storage_type> p_out_edges;

        typedef boost::mpl::vector<p_orientation_of_normal, p_dual_edge_length, p_weights, p_dual_edge_length_reciprocal, p_in_cells, p_out_edges>
                accessor_list_t;

        gridtools::domain_type<accessor_list_t> domain(
                boost::fusion::make_vector(&orientation_of_normal, &dual_edge_length, &weights,
                                           &dual_edge_length_reciprocal, &in_cells, &out_edges)
        );

        auto stencil_prep = gridtools::make_computation<backend_t>(
                domain,
                grid_,
                gridtools::make_mss // mss_descriptor
                        (execute<forward>(),
                         gridtools::make_esf<grad_n_prep, icosahedral_topology_t, icosahedral_topology_t::edges>(
                                 p_orientation_of_normal(), p_dual_edge_length(), p_weights(),
                                 p_dual_edge_length_reciprocal())
                        )
        );

        stencil_prep->ready();
        stencil_prep->steady();
        stencil_prep->run();

        /*
         *  stencil of flow convention
         */

        auto stencil_flow_convention = gridtools::make_computation<backend_t>(
                domain,
                grid_,
                gridtools::make_mss // mss_descriptor
                        (execute<forward>(),
                         gridtools::make_esf<grad_n_flow_convention, icosahedral_topology_t, icosahedral_topology_t::edges>(
                                 p_in_cells(), p_dual_edge_length(), p_out_edges())
                        )
        );

        stencil_flow_convention->ready();
        stencil_flow_convention->steady();
        stencil_flow_convention->run();

#ifdef __CUDACC__
        in_cells.d2h_update();
        dual_edge_length.d2h_update();
        out_edges.d2h_update();
#endif

        bool result = ver.verify(grid_, ref_edges, out_edges, halos);

#ifdef BENCHMARK
        for (uint_t t = 1; t < t_steps; ++t) {
            stencil_flow_convention->run();
        }
        stencil_flow_convention->finalize();
        std::cout << "flow convention: " << stencil_flow_convention->print_meter() << std::endl;
#endif

        /*
         * flow_convention_with_constexpr
         */

        auto stencil_flow_convention_with_constexpr = gridtools::make_computation<backend_t>(
                domain,
                grid_,
                gridtools::make_mss // mss_descriptor
                        (execute<forward>(),
                         gridtools::make_esf<grad_n_flow_convention_with_constexpr, icosahedral_topology_t, icosahedral_topology_t::edges>(
                                 p_in_cells(), p_dual_edge_length(), p_out_edges())
                        )
        );

        stencil_flow_convention_with_constexpr->ready();
        stencil_flow_convention_with_constexpr->steady();
        stencil_flow_convention_with_constexpr->run();

#ifdef __CUDACC__
        in_cells.d2h_update();
        dual_edge_length.d2h_update();
        out_edges.d2h_update();
#endif

        result = result && ver.verify(grid_, ref_edges, out_edges, halos);

#ifdef BENCHMARK
        for (uint_t t = 1; t < t_steps; ++t) {
            stencil_flow_convention_with_constexpr->run();
        }
        stencil_flow_convention_with_constexpr->finalize();
        std::cout << "flow convention with constexpr: " << stencil_flow_convention_with_constexpr->print_meter() <<
        std::endl;
#endif
        /*
         *  stencil of flow convention reciprocal
         */

        auto stencil_flow_convention_reciprocal = gridtools::make_computation<backend_t>(
                domain,
                grid_,
                gridtools::make_mss // mss_descriptor
                        (execute<forward>(),
                         gridtools::make_esf<grad_n_flow_convention_reciprocal, icosahedral_topology_t, icosahedral_topology_t::edges>(
                                 p_in_cells(), p_dual_edge_length_reciprocal(), p_out_edges())
                        )
        );

        stencil_flow_convention_reciprocal->ready();
        stencil_flow_convention_reciprocal->steady();
        stencil_flow_convention_reciprocal->run();

#ifdef __CUDACC__
        in_cells.d2h_update();
        dual_edge_length_reciprocal.d2h_update();
        out_edges.d2h_update();
#endif

        result = result && ver.verify(grid_, ref_edges, out_edges, halos);

#ifdef BENCHMARK
        for (uint_t t = 1; t < t_steps; ++t) {
            stencil_flow_convention_reciprocal->run();
        }
        stencil_flow_convention_reciprocal->finalize();
        std::cout << "flow convention with reciprocal: " << stencil_flow_convention_reciprocal->print_meter() <<
        std::endl;
#endif

        /*
         * flow_convention_three_esfs
         */

        auto stencil_flow_convention_three_esfs = gridtools::make_computation<backend_t>(
                domain,
                grid_,
                gridtools::make_mss // mss_descriptor
                        (execute<forward>(),
                         gridtools::make_cesf<0, grad_n_flow_convention_three_esfs<0>, icosahedral_topology_t, icosahedral_topology_t::edges>(
                                 p_in_cells(), p_dual_edge_length(), p_out_edges()),
                         gridtools::make_cesf<1, grad_n_flow_convention_three_esfs<1>, icosahedral_topology_t, icosahedral_topology_t::edges>(
                                 p_in_cells(), p_dual_edge_length(), p_out_edges()),
                         gridtools::make_cesf<2, grad_n_flow_convention_three_esfs<2>, icosahedral_topology_t, icosahedral_topology_t::edges>(
                                 p_in_cells(), p_dual_edge_length(), p_out_edges())
                        )
        );

        stencil_flow_convention_three_esfs->ready();
        stencil_flow_convention_three_esfs->steady();
        stencil_flow_convention_three_esfs->run();

#ifdef __CUDACC__
        in_cells.d2h_update();
        dual_edge_length.d2h_update();
        out_edges.d2h_update();
#endif

        result = result && ver.verify(grid_, ref_edges, out_edges, halos);

#ifdef BENCHMARK
        for (uint_t t = 1; t < t_steps; ++t) {
            stencil_flow_convention_three_esfs->run();
        }
        stencil_flow_convention_three_esfs->finalize();
        std::cout << "flow convention three esfs: " << stencil_flow_convention_three_esfs->print_meter() << std::endl;
#endif

        /*
         * stencil of orientation of normal
         */

        auto stencil_orientation_of_normal = gridtools::make_computation<backend_t>(
                domain,
                grid_,
                gridtools::make_mss // mss_descriptor
                        (execute<forward>(),
                         gridtools::make_esf<grad_n_orientation_of_normal, icosahedral_topology_t, icosahedral_topology_t::edges>(
                                 p_in_cells(), p_weights(), p_out_edges())
                        )
        );

        stencil_orientation_of_normal->ready();
        stencil_orientation_of_normal->steady();
        stencil_orientation_of_normal->run();

#ifdef __CUDACC__
        in_cells.d2h_update();
        weights.d2h_update();
        out_edges.d2h_update();
#endif

        result = result && ver.verify(grid_, ref_edges, out_edges, halos);

#ifdef BENCHMARK
        for (uint_t t = 1; t < t_steps; ++t) {
            stencil_orientation_of_normal->run();
        }
        stencil_orientation_of_normal->finalize();
        std::cout << "orientation of normal: " << stencil_orientation_of_normal->print_meter() << std::endl;
#endif

        return result;
    }
}


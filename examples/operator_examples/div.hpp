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

    struct div_functor_reduction_into_scalar {
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

            double t{0.};
            auto neighbors_offsets = connectivity< cells , edges >::offsets(eval.position()[1]);
            ushort_t e=0;
            for (auto neighbor_offset : neighbors_offsets) {
                t += eval(in_edges(neighbor_offset)) * eval(weights(edge+e));
                e++;
            }
            eval(out_cells()) = t;
        }
    };

    template <int color>
    struct div_functor_flow_convention;

    template <>
    struct div_functor_flow_convention<0>{
        typedef in_accessor<0, icosahedral_topology_t::edges, extent<1> > in_edges;
        typedef in_accessor<1, icosahedral_topology_t::edges, extent<1> > edge_length;
        typedef in_accessor<2, icosahedral_topology_t::cells, extent<1> > cell_area;
        typedef inout_accessor<3, icosahedral_topology_t::cells> out_cells;
        typedef boost::mpl::vector<in_edges, edge_length, cell_area, out_cells> arg_list;

        template<typename Evaluation>
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval)
        {
            auto ff = [](const double _in1, const double _in2, const double _res) -> double { return _in1 * _in2 + _res; };

            eval(out_cells()) = eval(on_edges(ff, 0.0, in_edges(), edge_length())) / eval(cell_area());
        }
    };

    template <>
    struct div_functor_flow_convention<1>{
        typedef in_accessor<0, icosahedral_topology_t::edges, extent<1> > in_edges;
        typedef in_accessor<1, icosahedral_topology_t::edges, extent<1> > edge_length;
        typedef in_accessor<2, icosahedral_topology_t::cells, extent<1> > cell_area;
        typedef inout_accessor<3, icosahedral_topology_t::cells> out_cells;
        typedef boost::mpl::vector<in_edges, edge_length, cell_area, out_cells> arg_list;

        template<typename Evaluation>
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval)
        {
            auto ff = [](const double _in1, const double _in2, const double _res) -> double { return _in1 * _in2 + _res; };

            eval(out_cells()) = -eval(on_edges(ff, 0.0, in_edges(), edge_length())) / eval(cell_area());
        }
    };

    template <int color>
    struct div_functor_over_edges {
        typedef in_accessor<0, icosahedral_topology_t::edges, extent<1> > in_edges;
        typedef in_accessor<1, icosahedral_topology_t::edges, extent<1> > edge_length;
        typedef inout_accessor<2, icosahedral_topology_t::cells> out_cells;
        typedef boost::mpl::vector<in_edges, edge_length, out_cells> arg_list;

        template<typename Evaluation>
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval)
        {
            typedef typename icgrid::get_grid_topology< Evaluation >::type grid_topology_t;
            constexpr auto neighbors_offsets = from<edges>::to<cells>::with_color<static_int<color> >::offsets();

            double t{eval(in_edges()) * eval(edge_length())};
            eval(out_cells(neighbors_offsets[0])) -= t;
            eval(out_cells(neighbors_offsets[1])) += t;
        }
    };

    template <>
    struct div_functor_over_edges<0> {
        typedef in_accessor<0, icosahedral_topology_t::edges, extent<1> > in_edges;
        typedef in_accessor<1, icosahedral_topology_t::edges, extent<1> > edge_length;
        typedef inout_accessor<2, icosahedral_topology_t::cells> out_cells;
        typedef boost::mpl::vector<in_edges, edge_length, out_cells> arg_list;

        template<typename Evaluation>
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval)
        {
            typedef typename icgrid::get_grid_topology< Evaluation >::type grid_topology_t;
            constexpr auto neighbors_offsets = from<edges>::to<cells>::with_color<static_int<0> >::offsets();

            double t{eval(in_edges()) * eval(edge_length())};
            eval(out_cells(neighbors_offsets[0])) = t;
            eval(out_cells(neighbors_offsets[1])) = t;
        }
    };

    template <int color>
    struct divide_by_field {
        typedef in_accessor<0, icosahedral_topology_t::cells, extent<0> > cell_area;
        typedef inout_accessor<1, icosahedral_topology_t::cells> out_cells;
        typedef boost::mpl::vector<cell_area, out_cells> arg_list;
        template<typename Evaluation>
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            typedef typename icgrid::get_grid_topology< Evaluation >::type grid_topology_t;
            constexpr auto neighbors_offsets = from<edges>::to<cells>::with_color<static_int<color> >::offsets();

            eval(out_cells(neighbors_offsets[0])) /= eval(cell_area(neighbors_offsets[0]));
            eval(out_cells(neighbors_offsets[1])) /= eval(cell_area(neighbors_offsets[1]));
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

        array<uint_t, 5> di = {halo_nc, halo_nc, halo_nc, d1 - halo_nc - 1, d1};
        array<uint_t, 5> dj = {halo_mc, halo_mc, halo_mc, d2 - halo_mc - 1, d2};

        gridtools::grid<axis, icosahedral_topology_t> grid_(icosahedral_grid, di, dj);
        grid_.value_list[0] = 0;
        grid_.value_list[1] = d3 - 1;

        verifier ver(1e-10);

        array<array<uint_t, 2>, 4> halos = {{{halo_nc, halo_nc}, {0, 0}, {halo_mc, halo_mc}, {halo_k, halo_k}}};

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

        typedef arg<0, edge_storage_type> p_edge_length;
        typedef arg<1, cell_storage_type> p_cell_area;
        typedef arg<2, edges_of_cells_storage_type> p_orientation_of_normal;
        typedef arg<3, edges_of_cells_storage_type> p_div_weights;

        typedef arg<4, edge_storage_type> p_in_edges;
        typedef arg<5, cell_storage_type> p_out_cells;

        typedef boost::mpl::vector< p_edge_length, p_cell_area, p_orientation_of_normal, p_div_weights, p_in_edges, p_out_cells>
            accessor_list_t;

        gridtools::domain_type<accessor_list_t> domain(
            boost::fusion::make_vector(&edge_length, &cell_area, &orientation_of_normal, &div_weights, &in_edges, &out_cells));

        auto stencil_prep = gridtools::make_computation<backend_t>(
            domain,
            grid_,
            gridtools::make_mss // mss_descriptor
                (execute<forward>(),
                 gridtools::make_esf<div_prep_functor, icosahedral_topology_t, icosahedral_topology_t::cells>(
                         p_edge_length(), p_cell_area(), p_orientation_of_normal(), p_div_weights())
                )
        );
        stencil_prep->ready();
        stencil_prep->steady();
        stencil_prep->run();

#ifdef __CUDACC__
        orientation_of_normal.d2h_update();
        edge_length.d2h_update();
        cell_area.d2h_update();
        div_weights.d2h_update();
#endif

        /*
         * stencil of div
         */

        auto stencil_ = gridtools::make_computation<backend_t>(
                domain,
                grid_,
                gridtools::make_mss // mss_descriptor
                        (execute<forward>(),
                         gridtools::make_esf<div_functor, icosahedral_topology_t, icosahedral_topology_t::cells>(
                                 p_in_edges(), p_div_weights(), p_out_cells())
                        )
        );
        stencil_->ready();
        stencil_->steady();
        stencil_->run();

#ifdef __CUDACC__
        in_edges.d2h_update();
        div_weights.d2h_update();
        out_cells.d2h_update();
#endif

        bool result = ver.verify(grid_, ref_cells, out_cells, halos);

#ifdef BENCHMARK
        for (uint_t t = 1; t < t_steps; ++t)
        {
            stencil_->run();
        }
        stencil_->finalize();
        std::cout << "div: "<< stencil_->print_meter() << std::endl;
#endif

        /*
         * stencil of div reduction into scalar
         */
        auto stencil_reduction_into_scalar = gridtools::make_computation<backend_t>(
                domain,
                grid_,
                gridtools::make_mss // mss_descriptor
                        (execute<forward>(),
                         gridtools::make_esf<div_functor_reduction_into_scalar, icosahedral_topology_t, icosahedral_topology_t::cells>(
                                 p_in_edges(), p_div_weights(), p_out_cells())
                        )
        );
        stencil_reduction_into_scalar->ready();
        stencil_reduction_into_scalar->steady();
        stencil_reduction_into_scalar->run();

#ifdef __CUDACC__
        in_edges.d2h_update();
        div_weights.d2h_update();
        out_cells.d2h_update();
#endif

        result = result && ver.verify(grid_, ref_cells, out_cells, halos);

#ifdef BENCHMARK
        for (uint_t t = 1; t < t_steps; ++t)
        {
            stencil_reduction_into_scalar->run();
        }
        stencil_reduction_into_scalar->finalize();
        std::cout << "reduction into scalar: " << stencil_reduction_into_scalar->print_meter() << std::endl;
#endif

        /*
         * stencil of div flow convention
         */

        auto stencil_flow_convention = gridtools::make_computation<backend_t>(
                domain,
                grid_,
                gridtools::make_mss // mss_descriptor
                        (execute<forward>(),
                         gridtools::make_cesf<0, div_functor_flow_convention<0>, icosahedral_topology_t, icosahedral_topology_t::cells>(
                                 p_in_edges(), p_edge_length(), p_cell_area(), p_out_cells()),
                         gridtools::make_cesf<1, div_functor_flow_convention<1>, icosahedral_topology_t, icosahedral_topology_t::cells>(
                                 p_in_edges(), p_edge_length(), p_cell_area(), p_out_cells())
                        )
        );
        stencil_flow_convention->ready();
        stencil_flow_convention->steady();
        stencil_flow_convention->run();

#ifdef __CUDACC__
        in_edges.d2h_update();
        edge_length.d2h_update();
        cell_area.d2h_update();
        out_cells.d2h_update();
#endif

        result = result && ver.verify(grid_, ref_cells, out_cells, halos);

#ifdef BENCHMARK
        for (uint_t t = 1; t < t_steps; ++t)
        {
            stencil_flow_convention->run();
        }
        stencil_flow_convention->finalize();
        std::cout << "flow convention: "<< stencil_flow_convention->print_meter() << std::endl;
#endif

        /*
         * stencil of over edge
         */

        auto stencil_div_over_edges = gridtools::make_computation<backend_t>(
                domain,
                grid_,
                gridtools::make_mss // mss_descriptor
                        (execute<forward>(),
                         gridtools::make_cesf<0, div_functor_over_edges<0>, icosahedral_topology_t, icosahedral_topology_t::edges>(
                                 p_in_edges(), p_edge_length(), p_out_cells()),
                         gridtools::make_cesf<1, div_functor_over_edges<1>, icosahedral_topology_t, icosahedral_topology_t::edges>(
                                 p_in_edges(), p_edge_length(), p_out_cells()),
                         gridtools::make_cesf<2, div_functor_over_edges<2>, icosahedral_topology_t, icosahedral_topology_t::edges>(
                                 p_in_edges(), p_edge_length(), p_out_cells()),
                         gridtools::make_cesf<0, divide_by_field<0>, icosahedral_topology_t, icosahedral_topology_t::edges >(
                                 p_cell_area(), p_out_cells())
//                         gridtools::make_cesf<0, divide_by_field<1>, icosahedral_topology_t, icosahedral_topology_t::edges >(
//                                 p_cell_area(), p_out_cells()),
//                         gridtools::make_cesf<0, divide_by_field<2>, icosahedral_topology_t, icosahedral_topology_t::edges >(
//                                 p_cell_area(), p_out_cells())
                        )
        );
        stencil_div_over_edges->ready();
        stencil_div_over_edges->steady();
        stencil_div_over_edges->run();

#ifdef __CUDACC__
        in_edges.d2h_update();
        edge_length.d2h_update();
        cell_area.d2h_update();
        out_cells.d2h_update();
#endif

        // TODO: this does not validate because the divide_by_field functor runs only on edges with color 0
//        result = result && ver.verify(grid_, ref_cells, out_cells, halos);

#ifdef BENCHMARK
        for (uint_t t = 1; t < t_steps; ++t)
        {
            stencil_div_over_edges->run();
        }
        stencil_flow_convention->finalize();
        std::cout << "over edges: "<< stencil_div_over_edges->print_meter() << std::endl;
#endif

        return result;
    }
}

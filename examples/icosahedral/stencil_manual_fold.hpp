#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include "tools/verifier.hpp"
#include "unstructured_grid.hpp"

using namespace gridtools;
using namespace enumtype;
using namespace expressions;

namespace smf {

#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

    using backend_t = BACKEND;
    using icosahedral_topology_t = ::gridtools::icosahedral_topology< backend_t >;

    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_interval;
    typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;

    struct test_on_edges_functor {
        typedef in_accessor< 0, icosahedral_topology_t::cells, extent< 1 > > cell_area;
        typedef inout_accessor< 1, icosahedral_topology_t::cells, 5 > weight_edges;
        typedef boost::mpl::vector< cell_area, weight_edges > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            typedef typename icgrid::get_grid_topology< Evaluation >::type grid_topology_t;

            using edge_of_cell_dim = dimension< 5 >;
            edge_of_cell_dim::Index edge;

            if (eval.position()[1] == 0) {
                constexpr auto neighbors_offsets = from< cells >::to< cells >::with_color< static_int< 0 > >::offsets();
                ushort_t e=0;
                for (auto neighbor_offset : neighbors_offsets) {

                    eval(weight_edges(edge+e)) = eval(cell_area(neighbor_offset)) / eval(cell_area());
                    e++;
                }
            } else if (eval.position()[1] == 1) {
                constexpr auto neighbors_offsets = from< cells >::to< cells >::with_color< static_int< 1 > >::offsets();
                ushort_t e=0;
                for (auto neighbor_offset : neighbors_offsets) {
                    eval(weight_edges(edge+e)) = eval(cell_area(neighbor_offset)) / eval(cell_area());
                    e++;
                }
            }
        }
    };

    bool test(uint_t x, uint_t y, uint_t z, uint_t t_steps) {

        uint_t d1 = x;
        uint_t d2 = y;
        uint_t d3 = z;

        typedef gridtools::layout_map< 2, 1, 0 > layout_t;

        using cell_storage_type = typename backend_t::storage_t< icosahedral_topology_t::cells, double >;

        const uint_t halo_nc = 1;
        const uint_t halo_mc = 1;
        const uint_t halo_k = 0;
        icosahedral_topology_t icosahedral_grid(d1, d2, d3);

        auto cell_area = icosahedral_grid.make_storage< icosahedral_topology_t::cells, double >("cell_area");
        auto weight_edges_meta = meta_storage_extender()(cell_area.meta_data(), 3);
        using edges_of_cells_storage_type =
            typename backend_t::storage_type< double, decltype(weight_edges_meta) >::type;
        edges_of_cells_storage_type weight_edges(weight_edges_meta, "edges_of_cell");
        edges_of_cells_storage_type ref_weights(weight_edges_meta, "ref_edges_of_cell");

        for (int i = 0; i < d1; ++i) {
            for (int c = 0; c < icosahedral_topology_t::cells::n_colors::value; ++c) {
                for (int j = 0; j < d2; ++j) {
                    for (int k = 0; k < d3; ++k) {
                        cell_area(i, c, j, k) = (uint_t)cell_area.meta_data().index(
                            array< uint_t, 4 >{(uint_t)i, (uint_t)c, (uint_t)j, (uint_t)k});
                    }
                }
            }
        }
        weight_edges.initialize(0.0);
        ref_weights.initialize(0.0);

        typedef arg< 0, cell_storage_type > p_cell_area;
        typedef arg< 1, edges_of_cells_storage_type > p_weight_edges;

        typedef boost::mpl::vector< p_cell_area, p_weight_edges > accessor_list_t;

        gridtools::domain_type< accessor_list_t > domain(boost::fusion::make_vector(&cell_area, &weight_edges));
        array< uint_t, 5 > di = {halo_nc, halo_nc, halo_nc, d1 - halo_nc - 1, d1};
        array< uint_t, 5 > dj = {halo_mc, halo_mc, halo_mc, d2 - halo_mc - 1, d2};

        gridtools::grid< axis, icosahedral_topology_t > grid_(icosahedral_grid, di, dj);
        grid_.value_list[0] = 0;
        grid_.value_list[1] = d3 - 1;

        auto stencil_ = gridtools::make_computation< backend_t >(
            domain,
            grid_,
            gridtools::make_mss // mss_descriptor
            (execute< forward >(),
                gridtools::make_esf< test_on_edges_functor, icosahedral_topology_t, icosahedral_topology_t::cells >(
                    p_cell_area(), p_weight_edges())));
        stencil_->ready();
        stencil_->steady();
        stencil_->run();

#ifdef __CUDACC__
        out_edges.d2h_update();
        in_edges.d2h_update();
#endif

        unstructured_grid ugrid(d1, d2, d3);
        for (uint_t i = halo_nc; i < d1 - halo_nc; ++i) {
            for (uint_t c = 0; c < icosahedral_topology_t::edges::n_colors::value; ++c) {
                for (uint_t j = halo_mc; j < d2 - halo_mc; ++j) {
                    for (uint_t k = 0; k < d3; ++k) {

                        auto neighbours =
                            ugrid.neighbours_of< icosahedral_topology_t::cells, icosahedral_topology_t::cells >(
                                {i, c, j, k});
                        ushort_t e=0;
                        for (auto iter = neighbours.begin(); iter != neighbours.end(); ++iter) {
                            ref_weights(i, c, j, k, e) = cell_area(*iter) / cell_area(i,c,j,k);
                            ++e;
                        }

                    }
                }
            }
        }

        verifier ver(1e-10);

        array< array< uint_t, 2 >, 5 > halos = {
            {{halo_nc, halo_nc}, {0, 0}, {halo_mc, halo_mc}, {halo_k, halo_k}, {0, 0}}};
        bool result = ver.verify(grid_, ref_weights, weight_edges, halos);

#ifdef BENCHMARK
        for (uint_t t = 1; t < t_steps; ++t) {
            stencil_->run();
        }
        stencil_->finalize();
        std::cout << stencil_->print_meter() << std::endl;
#endif

        return result;
    }
} // namespace soeov

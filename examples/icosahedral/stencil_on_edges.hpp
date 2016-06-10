#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include "tools/verifier.hpp"
#include "unstructured_grid.hpp"

using namespace gridtools;
using namespace enumtype;

namespace soe {

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
        typedef in_accessor< 0, icosahedral_topology_t::edges, extent< 1 > > in;
        typedef inout_accessor< 1, icosahedral_topology_t::edges > out;
        typedef in_accessor< 2, icosahedral_topology_t::edges, extent< 1 > > ipos;
        typedef in_accessor< 3, icosahedral_topology_t::edges, extent< 1 > > cpos;
        typedef in_accessor< 4, icosahedral_topology_t::edges, extent< 1 > > jpos;
        typedef in_accessor< 5, icosahedral_topology_t::edges, extent< 1 > > kpos;
        typedef boost::mpl::vector6< in, out, ipos, cpos, jpos, kpos > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            auto ff = [](const double _in, const double _res) -> double { return _in + _res; };

            /**
               This interface checks that the location types are compatible with the accessors
             */
            eval(out()) = eval(on_edges(ff, 0.0, in()));
        }
    };

    bool test(uint_t x, uint_t y, uint_t z, uint_t t_steps) {

        uint_t d1 = x;
        uint_t d2 = y;
        uint_t d3 = z;

        typedef gridtools::layout_map< 2, 1, 0 > layout_t;

        using edge_storage_type = typename backend_t::storage_t< icosahedral_topology_t::edges, double >;

        const uint_t halo_nc = 1;
        const uint_t halo_mc = 1;
        const uint_t halo_k = 0;
        icosahedral_topology_t icosahedral_grid(d1, d2, d3);

        auto in_edges = icosahedral_grid.make_storage< icosahedral_topology_t::edges, double >("in");
        auto i_edges = icosahedral_grid.make_storage< icosahedral_topology_t::edges, double >("i");
        auto j_edges = icosahedral_grid.make_storage< icosahedral_topology_t::edges, double >("j");
        auto c_edges = icosahedral_grid.make_storage< icosahedral_topology_t::edges, double >("c");
        auto k_edges = icosahedral_grid.make_storage< icosahedral_topology_t::edges, double >("k");
        auto out_edges = icosahedral_grid.make_storage< icosahedral_topology_t::edges, double >("out");
        auto ref_edges = icosahedral_grid.make_storage< icosahedral_topology_t::edges, double >("ref");

        for (int i = 0; i < d1; ++i) {
            for (int c = 0; c < icosahedral_topology_t::edges::n_colors::value; ++c) {
                for (int j = 0; j < d2; ++j) {
                    for (int k = 0; k < d3; ++k) {
                        in_edges(i, c, j, k) = (uint_t)in_edges.meta_data().index(
                            array< uint_t, 4 >{(uint_t)i, (uint_t)c, (uint_t)j, (uint_t)k});
                        i_edges(i, c, j, k) = i;
                        c_edges(i, c, j, k) = c;
                        j_edges(i, c, j, k) = j;
                        k_edges(i, c, j, k) = k;
                    }
                }
            }
        }
        out_edges.initialize(0.0);
        ref_edges.initialize(0.0);

        typedef arg< 0, edge_storage_type > p_in_edges;
        typedef arg< 1, edge_storage_type > p_out_edges;
        typedef arg< 2, edge_storage_type > p_i_edges;
        typedef arg< 3, edge_storage_type > p_c_edges;
        typedef arg< 4, edge_storage_type > p_j_edges;
        typedef arg< 5, edge_storage_type > p_k_edges;

        typedef boost::mpl::vector6< p_in_edges, p_out_edges, p_i_edges, p_c_edges, p_j_edges, p_k_edges >
            accessor_list_t;

        gridtools::domain_type< accessor_list_t > domain(
            boost::fusion::make_vector(&in_edges, &out_edges, &i_edges, &c_edges, &j_edges, &k_edges));
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
                gridtools::make_esf< test_on_edges_functor, icosahedral_topology_t, icosahedral_topology_t::edges >(
                    p_in_edges(), p_out_edges(), p_i_edges(), p_c_edges(), p_j_edges(), p_k_edges())));
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
                            ugrid.neighbours_of< icosahedral_topology_t::edges, icosahedral_topology_t::edges >(
                                {i, c, j, k});
                        for (auto iter = neighbours.begin(); iter != neighbours.end(); ++iter) {
                            ref_edges(i, c, j, k) += in_edges(*iter);
                        }
                    }
                }
            }
        }

        verifier ver(1e-10);

        array< array< uint_t, 2 >, 4 > halos = {{{halo_nc, halo_nc}, {0, 0}, {halo_mc, halo_mc}, {halo_k, halo_k}}};
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
} // namespace soe

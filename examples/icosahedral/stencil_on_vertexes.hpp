#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include "tools/verifier.hpp"
#include "unstructured_grid.hpp"
#include "../benchmarker.hpp"

using namespace gridtools;
using namespace enumtype;

namespace sov {

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

    template < uint_t Color >
    struct test_on_vertexes_functor {
        typedef in_accessor< 0, icosahedral_topology_t::vertexes, extent< 1 > > in;
        typedef inout_accessor< 1, icosahedral_topology_t::vertexes > out;
        typedef boost::mpl::vector2< in, out > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            auto ff = [](const double _in, const double _res) -> double { return _in + _res; };

            /**
               This interface checks that the location types are compatible with the accessors
             */
            eval(out()) = eval(on_vertexes(ff, 0.0, in()));
        }
    };

    bool test(uint_t x, uint_t y, uint_t z, uint_t t_steps, bool verify) {

        uint_t d1 = x;
        uint_t d2 = y;
        uint_t d3 = z;

        using cell_storage_type = typename backend_t::storage_t< icosahedral_topology_t::vertexes, double >;

        const uint_t halo_nc = 1;
        const uint_t halo_mc = 1;
        const uint_t halo_k = 0;
        icosahedral_topology_t icosahedral_grid(d1, d2, d3);

        auto in_vertexes = icosahedral_grid.make_storage< icosahedral_topology_t::vertexes, double >("in");
        auto out_vertexes = icosahedral_grid.make_storage< icosahedral_topology_t::vertexes, double >("out");
        auto ref_vertexes = icosahedral_grid.make_storage< icosahedral_topology_t::vertexes, double >("ref");

        for (int i = 0; i < d1; ++i) {
            for (int c = 0; c < icosahedral_topology_t::vertexes::n_colors::value; ++c) {
                for (int j = 0; j < d2; ++j) {
                    for (int k = 0; k < d3; ++k) {
                        in_vertexes(i, c, j, k) = (uint_t)in_vertexes.meta_data().index(
                            array< uint_t, 4 >{(uint_t)i, (uint_t)c, (uint_t)j, (uint_t)k});
                    }
                }
            }
        }
        out_vertexes.initialize(0.0);
        ref_vertexes.initialize(0.0);

        typedef arg< 0, cell_storage_type > p_in_vertexes;
        typedef arg< 1, cell_storage_type > p_out_vertexes;

        typedef boost::mpl::vector< p_in_vertexes, p_out_vertexes > accessor_list_t;

        gridtools::domain_type< accessor_list_t > domain(boost::fusion::make_vector(&in_vertexes, &out_vertexes));
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
                gridtools::make_esf< test_on_vertexes_functor,
                    icosahedral_topology_t,
                    icosahedral_topology_t::vertexes >(p_in_vertexes(), p_out_vertexes())));
        stencil_->ready();
        stencil_->steady();
        stencil_->run();

#ifdef __CUDACC__
        out_vertexes.d2h_update();
        in_vertexes.d2h_update();
#endif

        bool result = true;
        if (verify) {
            unstructured_grid ugrid(d1, d2, d3);
            for (uint_t i = halo_nc; i < d1 - halo_nc; ++i) {
                for (uint_t c = 0; c < icosahedral_topology_t::vertexes::n_colors::value; ++c) {
                    for (uint_t j = halo_mc; j < d2 - halo_mc + 1; ++j) {
                        for (uint_t k = 0; k < d3; ++k) {
                            auto neighbours = ugrid.neighbours_of< icosahedral_topology_t::vertexes,
                                icosahedral_topology_t::vertexes >({i, c, j, k});
                            for (auto iter = neighbours.begin(); iter != neighbours.end(); ++iter) {
                                ref_vertexes(i, c, j, k) += in_vertexes(*iter);
                            }
                        }
                    }
                }
            }

            verifier ver(1e-10);

            array< array< uint_t, 2 >, 4 > halos = {{{halo_nc, halo_nc}, {0, 0}, {halo_mc, halo_mc}, {halo_k, halo_k}}};
            bool result = ver.verify(grid_, ref_vertexes, out_vertexes, halos);
        }
#ifdef BENCHMARK
        benchmarker::run(stencil_, t_steps);
#endif
        stencil_->finalize();
        return result;
    }

} // namespace sov

//
// Created by Xiaolin Guo on 19.04.16.
//

#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include "tools/verifier.hpp"
#include "../icosahedral/unstructured_grid.hpp"
#include "div_repository.hpp"

using namespace gridtools;
using namespace enumtype;

namespace divergence {

    typedef gridtools::interval< level<0, -1>, level<1, -1> > x_interval;
    typedef gridtools::interval< level<0, -2>, level<1, 1> > axis;

    struct div_functor {
        typedef in_accessor<0, icosahedral_topology_t::edges, extent<1> > in_edges;
        typedef inout_accessor<1, icosahedral_topology_t::cells> out_cells;
        typedef in_accessor<2, icosahedral_topology_t::cells, extent<1> > cell_area;
        typedef in_accessor<3, icosahedral_topology_t::cells, extent<1> > edge_sign;
        typedef in_accessor<4, icosahedral_topology_t::edges, extent<1> > edge_length;
        typedef boost::mpl::vector<in_edges, out_cells, cell_area, edge_sign, edge_length> arg_list;

        template<typename Evaluation>
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval)
        {
            auto ff = [](const double _in, const double _res) -> double { return _in + _res; };

            /**
               This interface checks that the location types are compatible with the accessors
             */
            eval(out_cells()) = eval(on_edges(ff, 0.0, in_edges())) / eval(cell_area());
        }
    };

    bool test(uint_t x, uint_t y, uint_t z, uint_t t_steps, char *mesh_file)
    {

        uint_t d1 = x;
        uint_t d2 = y;
        uint_t d3 = z;

        divergence::repository repository(mesh_file);

        typedef gridtools::layout_map<2, 1, 0> layout_t;

        using edge_storage_type = typename backend_t::storage_t<icosahedral_topology_t::edges, double>;
        using cell_storage_type = typename backend_t::storage_t<icosahedral_topology_t::cells, double>;

        const uint_t halo_nc = 1;
        const uint_t halo_mc = 1;
        const uint_t halo_k = 0;
        icosahedral_topology_t icosahedral_grid(d1, d2, d3);

        auto& in_edges = repository.u();
        auto& cell_area = repository.cell_area();
        auto edge_sign = icosahedral_grid.make_storage<icosahedral_topology_t::cells, double>("edge_sign");
        auto& edge_length = repository.edge_length();
        auto out_cells = icosahedral_grid.make_storage<icosahedral_topology_t::cells, double>("out");
        auto& ref_cells = repository.div_u_ref();

        out_cells.initialize(0.0);
//        ref_cells.initialize(0.0);

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
                     p_in_edges(), p_out_cells(), p_cell_area(), p_edge_sign(), p_edge_length())));
        stencil_->ready();
        stencil_->steady();
        stencil_->run();

    #ifdef __CUDACC__
        out_edges.d2h_update();
            in_edges.d2h_update();
    #endif

//        unstructured_grid ugrid(d1, d2, d3);
//        for (uint_t i = halo_nc; i < d1 - halo_nc; ++i)
//        {
//            for (uint_t c = 0; c < icosahedral_topology_t::cells::n_colors::value; ++c)
//            {
//                for (uint_t j = halo_mc; j < d2 - halo_mc; ++j)
//                {
//                    for (uint_t k = 0; k < d3; ++k)
//                    {
////                        sign(i,0,j,k) = 1;
////                        sign(i,1,j,k) = -1;
//                        auto neighbours =
//                            ugrid.neighbours_of<icosahedral_topology_t::cells, icosahedral_topology_t::edges>(
//                                {i, c, j, k});
//                        for (auto iter = neighbours.begin(); iter != neighbours.end(); ++iter)
//                        {
//                            ref_cells(i, c, j, k) += in_edges(*iter);
//                            ref_cells
//                        }
//                    }
//                }
//            }
//        }

//        verifier ver(1e-10);
//
//        array<array<uint_t, 2>, 4> halos = {{{halo_nc, halo_nc}, {0, 0}, {halo_mc, halo_mc}, {halo_k, halo_k}}};
//        bool result = ver.verify(grid_, ref_cells, out_cells, halos);
//
//    #ifdef BENCHMARK
//        for (uint_t t = 1; t < t_steps; ++t)
//        {
//            stencil_->run();
//        }
//        stencil_->finalize();
//        std::cout << stencil_->print_meter() << std::endl;
//    #endif
//
//        return result;
        return true;
    }
} // namespace divergence

/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include "tools/verifier.hpp"

using namespace gridtools;
using namespace enumtype;

namespace cs_test{

#ifdef __CUDACC__
    using backend_t = ::gridtools::backend< Cuda, GRIDBACKEND, Block >;
#else
#ifdef BACKEND_BLOCK
    using backend_t = ::gridtools::backend< Host, GRIDBACKEND, Block >;
#else
    using backend_t = ::gridtools::backend< Host, GRIDBACKEND, Naive >;
#endif
#endif

    using icosahedral_topology_t = ::gridtools::icosahedral_topology<backend_t>;

    typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;
    typedef gridtools::interval<level<0,-2>, level<1,1> > axis;

    template < uint_t Color >
    struct test_functor {
        typedef in_accessor< 0, icosahedral_topology_t::cells, extent< 1 > > in;
        typedef inout_accessor<1, icosahedral_topology_t::cells> out;
        typedef boost::mpl::vector2<in, out> arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            eval(out())= eval(in());
        }
    };
}

using namespace cs_test;

TEST(test_copy_stencil, run) {

    typedef gridtools::layout_map<2,1,0> layout_t;

    using cell_storage_type = typename backend_t::storage_t<icosahedral_topology_t::cells, double>;

    const uint_t halo_nc = 1;
    const uint_t halo_mc = 2;
    const uint_t halo_k = 0;
    const uint_t d3=6+halo_k*2;
    const uint_t d1=6+halo_nc*2;
    const uint_t d2=12+halo_mc*2;
    icosahedral_topology_t icosahedral_grid( d1, d2, d3 );

    cell_storage_type in_cells = icosahedral_grid.make_storage<icosahedral_topology_t::cells, double>("in");
    cell_storage_type out_cells = icosahedral_grid.make_storage<icosahedral_topology_t::cells, double>("out");

    for(int i=0; i < d1; ++i)
    {
        for(int c=0; c < 2; ++c)
        {
            for(int j=0; j < d2; ++j)
            {
                for(int k=0; k < d3; ++k)
                {
                    in_cells(i,c,j,k) = i+c*100+j*10000+k*1000000;
                }
            }
        }
    }

    out_cells.initialize( 1.1 );

    typedef arg<0, cell_storage_type> p_in_cells;
    typedef arg<1, cell_storage_type> p_out_cells;

    typedef boost::mpl::vector<p_in_cells, p_out_cells> accessor_list_t;

    gridtools::aggregator_type<accessor_list_t> domain(boost::fusion::make_vector(&in_cells, &out_cells) );
    array<uint_t,5> di = {halo_nc, halo_nc, halo_nc, d1 - halo_nc -1, d1};
    array<uint_t,5> dj = {halo_mc, halo_mc, halo_mc, d2 - halo_mc -1, d2};

    gridtools::grid<axis, icosahedral_topology_t> grid_(icosahedral_grid, di, dj);
    grid_.value_list[0] = halo_k;
    grid_.value_list[1] = d3 - 1 - halo_k;

#ifdef CXX11_ENABLED
    auto
#else
#ifdef __CUDACC__
    gridtools::stencil *
#else
    boost::shared_ptr< gridtools::stencil >
#endif
#endif
            copy = gridtools::make_computation<backend_t >
            (
                domain, grid_,
                gridtools::make_multistage // mss_descriptor
                (
                    execute<forward>(),
                    gridtools::make_stage<test_functor, icosahedral_topology_t, icosahedral_topology_t::cells>(
                        p_in_cells(), p_out_cells() )
                )
            );
    copy->ready();
    copy->steady();
    copy->run();

#ifdef __CUDACC__
    in_cells.d2h_update();
    out_cells.d2h_update();
#endif

    bool result = true;
    for (int i = halo_nc; i < d1 - halo_nc; ++i) {
        for (int c = 0; c < 2; ++c) {
            for (int j = halo_mc; j < d2 - halo_mc; ++j) {
                for (int k = 0; k < d3; ++k) {
                    if (in_cells(i, c, j, k) != out_cells(i, c, j, k)) {
                        std::cout << "ERRRRROR " << i << " " << c << " " << j << " " << k << " " << in_cells(i, c, j, k)
                                  << " " << out_cells(i, c, j, k) << std::endl;
                        result = false;
                    }
                }
            }
        }
    }

    // TODO recover verifier
    //    verifier ver(1e-10);

    //    array<array<uint_t, 2>, 4> halos = {{ {halo_nc, halo_nc},{0,0},{halo_mc, halo_mc},{halo_k, halo_k} }};
    //    EXPECT_TRUE(ver.verify(grid_, in_cells, out_cells, halos));
    EXPECT_TRUE(result);
}

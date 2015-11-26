#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include "tools/verifier.hpp"

using namespace gridtools;
using namespace enumtype;

namespace cs_test{

    using backend_t = ::gridtools::backend<Host, Naive >;
    using trapezoid_2D_t = ::gridtools::trapezoid_2D_colored<backend_t>;

    typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;
    typedef gridtools::interval<level<0,-2>, level<1,1> > axis;

    struct test_functor {
        typedef ro_accessor<0, trapezoid_2D_t::cells, radius<1> > in;
        typedef accessor<1, trapezoid_2D_t::cells> out;
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

    using cell_storage_type = typename backend_t::storage_t<trapezoid_2D_t::cells, double>;

    const uint_t halo_nc = 1;
    const uint_t halo_mc = 2;
    const uint_t halo_k = 0;
    const uint_t d3=6+halo_k*2;
    const uint_t d1=6+halo_nc*2;
    const uint_t d2=12+halo_mc*2;
    trapezoid_2D_t grid( d1, d2, d3 );

    cell_storage_type in_cells = grid.make_storage<trapezoid_2D_t::cells, double>("in");
    cell_storage_type out_cells = grid.make_storage<trapezoid_2D_t::cells, double>("out");

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

    gridtools::domain_type<accessor_list_t> domain(boost::fusion::make_vector(&in_cells, &out_cells) );
    array<uint_t,5> di = {halo_nc, halo_nc, halo_nc, d1 - halo_nc -1, d1};
    array<uint_t,5> dj = {halo_mc, halo_mc, halo_mc, d2 - halo_mc -1, d2};

    gridtools::coordinates<axis, trapezoid_2D_t> coords(grid, di, dj);
    coords.value_list[0] = 0;
    coords.value_list[1] = d3-1;

#ifdef __CUDACC__
        gridtools::computation* copy =
#else
            boost::shared_ptr<gridtools::computation> copy =
#endif
            gridtools::make_computation<backend_t >
            (
                gridtools::make_mss // mss_descriptor
                (
                    execute<forward>(),
                    gridtools::make_esf<test_functor, trapezoid_2D_t, trapezoid_2D_t::cells>(
                        p_in_cells(), p_out_cells() )
                ),
                domain, coords
            );
    copy->ready();
    copy->steady();
    copy->run();

    verifier ver(1e-10);

    array<array<uint_t, 2>, 4> halos = {{ {halo_nc, halo_nc},{0,0},{halo_mc, halo_mc},{halo_k, halo_k} }};
    EXPECT_TRUE(ver.verify(coords, in_cells, out_cells, halos));
}

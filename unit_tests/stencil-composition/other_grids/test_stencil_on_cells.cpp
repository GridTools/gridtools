#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include "tools/verifier.hpp"
#include "unstructured_grid.hpp"

using namespace gridtools;
using namespace enumtype;

namespace cs_test{

    using backend_t = ::gridtools::backend<Host, Naive >;
    using trapezoid_2D_t = ::gridtools::trapezoid_2D_colored<backend_t>;

    typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;
    typedef gridtools::interval<level<0,-2>, level<1,1> > axis;

    struct test_on_cells_functor {
        typedef ro_accessor<0, trapezoid_2D_t::cells, radius<1> > in;
        typedef accessor<1, trapezoid_2D_t::cells> out;
        typedef ro_accessor<2, trapezoid_2D_t::cells, radius<1> > ipos;
        typedef ro_accessor<3, trapezoid_2D_t::cells, radius<1> > cpos;
        typedef ro_accessor<4, trapezoid_2D_t::cells, radius<1> > jpos;
        typedef ro_accessor<5, trapezoid_2D_t::cells, radius<1> > kpos;
        typedef boost::mpl::vector6<in, out, ipos, cpos, jpos, kpos> arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            auto ff = [](const double _in, const double _res) -> double
                {
                return _in+_res;
                 };

            /**
               This interface checks that the location types are compatible with the accessors
             */
            eval(out()) = eval(on_cells(ff, 0.0, in()));
        }
    };
}

using namespace cs_test;

TEST(test_stencil_on_cells, run) {

    typedef gridtools::layout_map<2,1,0> layout_t;

    using cell_storage_type = typename backend_t::storage_t<trapezoid_2D_t::cells, double>;

    const uint_t halo_nc = 1;
    const uint_t halo_mc = 2;
    const uint_t halo_k = 0;
    const uint_t d3=6+halo_k*2;
    const uint_t d1=6+halo_nc*2;
    const uint_t d2=12+halo_mc*2;
    trapezoid_2D_t grid( d1, d2, d3 );

    cell_storage_type in_cells = grid.make_storage<trapezoid_2D_t::cells>("in");
    cell_storage_type i_cells = grid.make_storage<trapezoid_2D_t::cells>("i");
    cell_storage_type j_cells = grid.make_storage<trapezoid_2D_t::cells>("j");
    cell_storage_type c_cells = grid.make_storage<trapezoid_2D_t::cells>("c");
    cell_storage_type k_cells = grid.make_storage<trapezoid_2D_t::cells>("k");
    cell_storage_type out_cells = grid.make_storage<trapezoid_2D_t::cells>("out");
    cell_storage_type ref_cells = grid.make_storage<trapezoid_2D_t::cells>("ref");

    for(int i=0; i < d1; ++i)
    {
        for(int c=0; c < 2; ++c)
        {
            for(int j=0; j < d2; ++j)
            {
                for(int k=0; k < d3; ++k)
                {
                    in_cells(i,c,j,k) = i+c*100+j*10000+k*1000000;
                    i_cells(i,c,j,k) = i;
                    c_cells(i,c,j,k) = c;
                    j_cells(i,c,j,k) = j;
                    k_cells(i,c,j,k) = k;
                }
            }
        }
    }
    out_cells.initialize(0.0);
    ref_cells.initialize(0.0);

    typedef arg<0, cell_storage_type> p_in_cells;
    typedef arg<1, cell_storage_type> p_out_cells;
    typedef arg<2, cell_storage_type> p_i_cells;
    typedef arg<3, cell_storage_type> p_c_cells;
    typedef arg<4, cell_storage_type> p_j_cells;
    typedef arg<5, cell_storage_type> p_k_cells;

    typedef boost::mpl::vector<p_in_cells, p_out_cells, p_i_cells, p_c_cells, p_j_cells, p_k_cells> accessor_list_t;

    gridtools::domain_type<accessor_list_t> domain(boost::fusion::make_vector(&in_cells, &out_cells, &i_cells, &c_cells, &j_cells, &k_cells) );
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
                    gridtools::make_esf<test_on_cells_functor, trapezoid_2D_t, trapezoid_2D_t::cells>(
                        p_in_cells(), p_out_cells(), p_i_cells(), p_c_cells(), p_j_cells(), p_k_cells() )
                ),
                domain, coords
            );
    copy->ready();
    copy->steady();
    copy->run();

    unstructured_grid ugrid(d1, d2, d3);
    for(uint_t i=0; i < d1; ++i)
    {
        for(uint_t c=0; c < 2; ++c)
        {
            for(uint_t j=0; j < d2; ++j)
            {
                for(uint_t k=0; k < d3; ++k)
                {
                    auto neighbours = ugrid.neighbours_of({i,c,j,k});
                    for(auto iter = neighbours.begin(); iter != neighbours.end(); ++iter)
                    {
                        ref_cells(i,c,j,k) += in_cells(*iter);
                    }
                }
            }
        }
    }

    verifier ver(1e-10);

    array<array<uint_t, 2>, 4> halos = {{halo_nc, halo_nc},{0,0},{halo_mc, halo_mc},{halo_k, halo_k}};
    EXPECT_TRUE(ver.verify(ref_cells, out_cells, halos));
}

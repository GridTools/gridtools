#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <stencil-composition/stencil-composition.hpp>

using namespace gridtools;
using namespace enumtype;

namespace cs_test{

    using backend_t = ::gridtools::backend<Host, Block >;
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

    const uint_t d3=6;
    const uint_t d1=6;
    const uint_t d2=12;
    trapezoid_2D_t grid( d1, d2, d3 );
    cell_storage_type in_cells = grid.make_storage<trapezoid_2D_t::cells>();
    cell_storage_type out_cells = grid.make_storage<trapezoid_2D_t::cells>();


    typedef arg<0, cell_storage_type> p_in_cells;
    typedef arg<1, cell_storage_type> p_out_cells;

    typedef boost::mpl::vector<p_in_cells, p_out_cells> accessor_list_t;

    gridtools::domain_type<accessor_list_t> domain(boost::fusion::make_vector(&in_cells, &out_cells) );
    uint_t di[5] = {0, 0, 0, d1-1, d1};
    uint_t dj[5] = {0, 0, 0, d2-1, d2};

    gridtools::coordinates<axis> coords(di, dj);
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


    typedef decltype(
        gridtools::make_mss // mss_descriptor
        (
                enumtype::execute<enumtype::forward>(),
                gridtools::make_esf<test_functor, trapezoid_2D_t, trapezoid_2D_t::cells>(p_in_cells())
        )) mss1_t;

    typedef decltype(
        gridtools::make_mss // mss_descriptor
        (
                enumtype::execute<enumtype::forward>(),
                gridtools::make_esf<test_functor, trapezoid_2D_t, trapezoid_2D_t::cells>(p_out_cells())
        )) mss2_t;

    typedef gridtools::interval<level<0,-2>, level<1,1> > axis_t;
    typedef gridtools::coordinates<axis_t> coords_t;

    typedef gridtools::domain_type<accessor_list_t> domain_t;
    typedef boost::mpl::vector5<int, domain_t, mss2_t, coords_t, mss1_t> ListTypes;

    typedef _impl::get_mss_array<ListTypes>::type MssArray;

    BOOST_STATIC_ASSERT(( boost::mpl::equal<MssArray::elements, boost::mpl::vector2<mss2_t, mss1_t> >::value));
    EXPECT_TRUE(true);
}

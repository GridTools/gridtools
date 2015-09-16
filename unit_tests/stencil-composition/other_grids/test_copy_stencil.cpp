#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <stencil-composition/stencil-composition.hpp>

using namespace gridtools;

namespace make_computation_test{

    typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;

    struct test_functor {
        typedef ro_accessor<0, radius<1> > in;
        typedef boost::mpl::vector1<in> arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {}
    };
}

TEST(test_copy_stencil, run) {

    using namespace gridtools;

    using backend_t = backend<enumtype::Host, enumtype::Block >;

    typedef gridtools::layout_map<2,1,0> layout_t;
    using trapezoid_2D_t = gridtools::trapezoid_2D_colored<backend_t>;

    using cell_storage_type = typename backend_t::storage_t<trapezoid_2D_t::cells, double>;

    typedef arg<0, trapezoid_2D_t::cells> in_cells;
    typedef arg<1, trapezoid_2D_t::cells> out_cells;

    typedef boost::mpl::vector<in_cells, out_cells> accessor_list_t;

    typedef decltype(
        gridtools::make_mss // mss_descriptor
        (
                enumtype::execute<enumtype::forward>(),
                gridtools::make_esf<make_computation_test::test_functor, trapezoid_2D_t, trapezoid_2D_t::cells>(in_cells())
        )) mss1_t;

    typedef decltype(
        gridtools::make_mss // mss_descriptor
        (
                enumtype::execute<enumtype::forward>(),
                gridtools::make_esf<make_computation_test::test_functor, trapezoid_2D_t, trapezoid_2D_t::cells>(out_cells())
        )) mss2_t;

    typedef gridtools::interval<level<0,-2>, level<1,1> > axis_t;
    typedef gridtools::coordinates<axis_t> coords_t;

    typedef gridtools::domain_type<accessor_list_t> domain_t;
    typedef boost::mpl::vector5<int, domain_t, mss2_t, coords_t, mss1_t> ListTypes;

    typedef _impl::get_mss_array<ListTypes>::type MssArray;

    BOOST_STATIC_ASSERT(( boost::mpl::equal<MssArray::elements, boost::mpl::vector2<mss2_t, mss1_t> >::value));
    EXPECT_TRUE(true);
}

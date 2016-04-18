#include <stencil-composition/stencil-composition.hpp>
#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <stencil-composition/structured_grids/compute_extents_metafunctions.hpp>

using namespace gridtools;
using namespace gridtools::enumtype;
using gridtools::accessor;
using gridtools::extent;
using gridtools::layout_map;
using gridtools::float_type;
using gridtools::arg;
using gridtools::uint_t;
using gridtools::int_t;

typedef gridtools::interval<gridtools::level<0,-1>, gridtools::level<1,-1> > x_interval;
struct print_r {
    template <typename T>
    void operator()(T const& ) const {
        std::cout << typename T::first() << " " << typename T::second() << std::endl;
    }
};

struct functor0{
    typedef accessor<0, enumtype::in, extent<-1, 0, -2, 1, -3, 2> > in0;
    typedef accessor<1, enumtype::in, extent<-3, 2, -2, 0, 0, 2> > in1;
    typedef accessor<2, enumtype::inout> out;
    typedef accessor<3, enumtype::in, extent<-1, 2, 0, 0, -3, 1> > in3;

    typedef boost::mpl::vector<in0,in1,out,in3> arg_list;

    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_interval) {}
};


std::ostream& operator<<(std::ostream& s, functor0) { return s << "functor0"; }
#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Block >
#endif

typedef layout_map<2,1,0> layout_ijk_t;
typedef gridtools::BACKEND::storage_info<0, layout_ijk_t> storage_info_ijk_t;
typedef gridtools::BACKEND::storage_type<float_type, storage_info_ijk_t >::type storage_type;
typedef gridtools::BACKEND::temporary_storage_type<float_type, storage_info_ijk_t >::type tmp_storage_type;


typedef arg<0, storage_type> o0;
typedef arg<1, storage_type> in0;
typedef arg<2, storage_type> in1;
typedef arg<3, storage_type> in2;

TEST(esf_metafunctions, compute_extents_of)
{
    typedef decltype(gridtools::make_esf<functor0>(in0(), in1(), o0(), in2())) functor0__;
    typedef decltype( gridtools::make_mss
        (
            execute<forward>(),
            functor0__()        )
    ) mss_t;
    typedef boost::mpl::vector<o0, in0, in1, in2> placeholders;

    typedef gridtools::strgrid::compute_extents_of<placeholders>::for_mss<mss_t>::type final_map;

GRIDTOOLS_STATIC_ASSERT((std::is_same<boost::mpl::at<final_map, o0>::type, extent<0, 0, 0, 0, 0, 0> >::type::value),
                          "o0 extent<0, 0, 0, 0, 0, 0>");
GRIDTOOLS_STATIC_ASSERT((std::is_same<boost::mpl::at<final_map, in0>::type, extent<-1, 0, -2, 1, -3, 2> >::type::value),
                          "in0 extent<-1, 0, -2, 1, -3, 2>");
GRIDTOOLS_STATIC_ASSERT((std::is_same<boost::mpl::at<final_map, in1>::type, extent<-3, 2, -2, 0, 0, 2> >::type::value),
                          "in1 extent<-3, 2, -2, 0, 0, 2>");
GRIDTOOLS_STATIC_ASSERT((std::is_same<boost::mpl::at<final_map, in2>::type, extent<-1, 2, 0, 0, -3, 1> >::type::value),
                          "in2 extent<-1, 2, 0, 0, -3, 1>");
/* total placeholders (rounded to 10) _SIZE = 10*/
    ASSERT_TRUE(true);
}

namespace _for_test {
    template <int I>
    struct arg {
        typedef boost::mpl::int_<I> index_type;
    };
} // namespace _for_test

TEST(esf_metafunctions, check_arg_list_order)
{
    GRIDTOOLS_STATIC_ASSERT((gridtools::_impl::check_arg_list<boost::mpl::vector<_for_test::arg<0>,_for_test::arg<1>,_for_test::arg<2>,_for_test::arg<3> > >::value == true), "Test1 failed");

    GRIDTOOLS_STATIC_ASSERT((gridtools::_impl::check_arg_list<boost::mpl::vector<_for_test::arg<0>,_for_test::arg<4>,_for_test::arg<2>,_for_test::arg<3> > >::value == false), "Test1 failed");

    GRIDTOOLS_STATIC_ASSERT((gridtools::_impl::check_arg_list<boost::mpl::vector<_for_test::arg<0>,_for_test::arg<0>,_for_test::arg<3>,_for_test::arg<3> > >::value == false), "Test1 failed");
    ASSERT_TRUE(true);
}

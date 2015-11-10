#include "common/defs.hpp"
#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include "stencil-composition/backend.hpp"
#include "stencil-composition/caches/cache_metafunctions.hpp"
#include "stencil-composition/caches/define_caches.hpp"
#include "stencil-composition/interval.hpp"
#include "stencil-composition/make_computation.hpp"

using namespace gridtools;
using namespace gridtools::enumtype;
using gridtools::accessor;
using gridtools::range;
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
    typedef accessor<0, enumtype::in, range<-1, 0, -2, 1, -3, 2>> in0;
    typedef accessor<1, enumtype::in, range<-3, 2, -2, 0, 0, 2>> in1;
    typedef accessor<2, enumtype::inout> out;
    typedef accessor<3, enumtype::in, range<-1, 2, 0, 0, -3, 1>> in3;

    typedef boost::mpl::vector<in0,in1,out,in3> arg_list;

    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_interval) {}
};


std::ostream& operator<<(std::ostream& s, functor0) { return s << "functor0"; }
#ifdef __CUDACC__
  #define BACKEND backend<Cuda, Block >
#else
  #define BACKEND backend<Host, Block >
#endif

typedef layout_map<2,1,0> layout_ijk_t;
typedef storage_info<0, layout_ijk_t> storage_info_ijk_t;
typedef gridtools::BACKEND::storage_type<float_type, storage_info_ijk_t >::type storage_type;
typedef gridtools::BACKEND::temporary_storage_type<float_type, storage_info_ijk_t >::type tmp_storage_type;


typedef arg<0, storage_type> o0;
typedef arg<1, storage_type> in0;
typedef arg<2, storage_type> in1;
typedef arg<3, storage_type> in2;

TEST(esf_metafunctions, compute_ranges_of)
{
    typedef decltype(gridtools::make_esf<functor0>(in0(), in1(), o0(), in2())) functor0__;
    typedef decltype( gridtools::make_mss
        (
            execute<forward>(),
            functor0__()        )
    ) mss_t;
    typedef boost::mpl::vector<o0, in0, in1, in2> placeholders;

    typedef gridtools::compute_ranges_of<placeholders>::for_mss<mss_t>::type final_map;

GRIDTOOLS_STATIC_ASSERT((std::is_same<boost::mpl::at<final_map, o0>::type, range<0, 0, 0, 0, 0, 0>>::type::value),
                          "o0 range<0, 0, 0, 0, 0, 0>");
GRIDTOOLS_STATIC_ASSERT((std::is_same<boost::mpl::at<final_map, in0>::type, range<-1, 0, -2, 1, -3, 2>>::type::value),
                          "in0 range<-1, 0, -2, 1, -3, 2>");
GRIDTOOLS_STATIC_ASSERT((std::is_same<boost::mpl::at<final_map, in1>::type, range<-3, 2, -2, 0, 0, 2>>::type::value),
                          "in1 range<-3, 2, -2, 0, 0, 2>");
GRIDTOOLS_STATIC_ASSERT((std::is_same<boost::mpl::at<final_map, in2>::type, range<-1, 2, 0, 0, -3, 1>>::type::value),
                          "in2 range<-1, 2, 0, 0, -3, 1>");
/* total placeholders (rounded to 10) _SIZE = 10*/
    ASSERT_TRUE(true);
}

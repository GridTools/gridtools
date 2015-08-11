#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include "common/defs.hpp"
#include "stencil-composition/backend.hpp"
#include "stencil-composition/caches/cache_metafunctions.hpp"
#include "stencil-composition/caches/define_caches.hpp"
#include "stencil-composition/interval.hpp"
#include "stencil-composition/make_computation.hpp"



using namespace gridtools::enumtype;
using gridtools::accessor;
using gridtools::range;
using gridtools::layout_map;
using gridtools::float_type;
using gridtools::arg;
using gridtools::uint_t;
using gridtools::int_t;



// This is the definition of the special regions in the "vertical" direction
typedef gridtools::interval<gridtools::level<0,-1>, gridtools::level<1,-1> > x_interval;

struct functor1 {
    typedef const accessor<0, range<-3,4,0,0,0,0> > in;
    typedef accessor<1> out;
    typedef boost::mpl::vector<in,out> arg_list;

    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_interval) {}
};

struct functor2 {
    typedef const accessor<0, range<-1,2,0,0,0,0> > z;
    typedef accessor<1> out;
    typedef boost::mpl::vector<z,out> arg_list;

    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_interval) {}
};

struct functor3 {
    typedef const accessor<0, range<0,5,0,0,0,0> > z;
    typedef const accessor<1, range<0,1,0,0,0,0> > y;
    typedef accessor<2> out;

    typedef boost::mpl::vector<z,y,out> arg_list;

    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_interval) {}
};

struct functor4 {
    typedef const accessor<0, range<-1,1,0,0,0,0> > x;
    typedef const accessor<1, range<-2,0,0,0,0,0> > y;
    typedef accessor<2> out;

    typedef boost::mpl::vector<x,y,out> arg_list;

    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_interval) {}
};

std::ostream& operator<<(std::ostream& s, functor1) { return s << "functor1"; }
std::ostream& operator<<(std::ostream& s, functor2) { return s << "functor2"; }
std::ostream& operator<<(std::ostream& s, functor3) { return s << "functor3"; }
std::ostream& operator<<(std::ostream& s, functor4) { return s << "functor4"; }

#ifdef __CUDACC__
  #define BACKEND backend<Cuda, Block >
#else
  #define BACKEND backend<Host, Block >
#endif

typedef layout_map<2,1,0> layout_ijk_t;
typedef gridtools::BACKEND::storage_type<float_type, layout_ijk_t >::type storage_type;
typedef gridtools::BACKEND::temporary_storage_type<float_type, layout_ijk_t >::type tmp_storage_type;

typedef arg<0, storage_type> p_x;
typedef arg<1, storage_type> p_y;
typedef arg<2, storage_type> p_z;
typedef arg<3, storage_type> p_i;
typedef arg<4, storage_type> p_o;

struct print_pair {
    template <typename Pair>
    void operator()(Pair const&) const {
        std::cout << typename Pair::first::index() << " -> " << typename Pair::second() << std::endl;
    }
};

template <typename MapArgsRanges, typename ESF>
struct check_pair {
    template <typename Index>
    void operator()(Index const&) const {
        typedef typename boost::mpl::at_c<typename ESF::args_t, Index::value>::type current_arg;
        typedef typename boost::mpl::at<MapArgsRanges, current_arg>::type range_from_esf;
        typedef typename boost::mpl::at_c<typename ESF::esf_function::arg_list, Index::value>::type::range_type range_from_functor;
        // std::cout << range_from_esf() << " == " << range_from_functor() << std::endl;
        GRIDTOOLS_STATIC_ASSERT((std::is_same<range_from_esf, range_from_functor>::value), "ERROR\nRanges do not match");
    }
};


struct print_r {
    template <typename T>
    void operator()(T const& ) const {
        std::cout << typename T::first() << " " << typename T::second() << std::endl;
    }
};

struct print_s {
    template <typename T>
    void operator()(T const& ) const {
        std::cout << T() << std::endl;
    }
};

TEST(esf_metafunctions, computing_rages)
{
    //storage_type in(10, 10, 10, 1.0, "in"), out(10, 10, 10, 1.0, "out");

    typedef decltype(gridtools::make_esf<functor1>(p_i(), p_z())) esf1_t;
    typedef decltype(gridtools::make_esf<functor2>(p_z(), p_y())) esf2_t;
    typedef decltype(gridtools::make_esf<functor3>(p_z(), p_y(), p_x())) esf3_t;
    typedef decltype(gridtools::make_esf<functor4>(p_x(), p_y(), p_o())) esf4_t;

    typedef decltype( gridtools::make_mss // mss_descriptor
        (
            execute<forward>(),
            esf1_t(), // esf_descriptor
            esf2_t(), // esf_descriptor
            esf3_t(), // esf_descriptor
            esf4_t() // esf_descriptor
        )
    ) mss_t;

    typedef boost::mpl::vector<p_i, p_o, p_x, p_y, p_z> placeholders;

    // std::cout << esf1_t() << ": \n";
    // boost::mpl::for_each<typename esf1_t::args_with_ranges>(print_pair());
    // std::cout << esf2_t() << ": \n";
    // boost::mpl::for_each<typename esf2_t::args_with_ranges>(print_pair());
    // std::cout << esf3_t() << ": \n";
    // boost::mpl::for_each<typename esf3_t::args_with_ranges>(print_pair());
    // std::cout << esf4_t() << ": \n";
    // boost::mpl::for_each<typename esf4_t::args_with_ranges>(print_pair());

    boost::mpl::for_each<typename boost::mpl::range_c
                         <uint_t,
                          0,
                          boost::mpl::size<typename esf1_t::args_with_ranges>::type::value> >
        (check_pair<typename esf1_t::args_with_ranges, esf1_t>());

    boost::mpl::for_each<typename boost::mpl::range_c
                         <uint_t,
                          0,
                          boost::mpl::size<typename esf2_t::args_with_ranges>::type::value> >
        (check_pair<typename esf2_t::args_with_ranges, esf2_t>());

    boost::mpl::for_each<typename boost::mpl::range_c
                         <uint_t,
                          0,
                          boost::mpl::size<typename esf3_t::args_with_ranges>::type::value> >
        (check_pair<typename esf3_t::args_with_ranges, esf3_t>());

    boost::mpl::for_each<typename boost::mpl::range_c
                         <uint_t,
                          0,
                          boost::mpl::size<typename esf4_t::args_with_ranges>::type::value> >
        (check_pair<typename esf4_t::args_with_ranges, esf4_t>());

    typedef gridtools::pass_temps<placeholders>::mss_compute_range_sizes_new<mss_t>::type final_map;

    std::cout << "FINAL" << std::endl;
    boost::mpl::for_each<final_map>(print_r());

    GRIDTOOLS_STATIC_ASSERT((std::is_same<boost::mpl::at<final_map, p_x>::type, range<-1,1,0,0,0,0>>::type::value),
                            "p_x range is wrong");
    GRIDTOOLS_STATIC_ASSERT((std::is_same<boost::mpl::at<final_map, p_y>::type, range<-2,2,0,0,0,0>>::type::value),
                            "p_y range is wrong");
    GRIDTOOLS_STATIC_ASSERT((std::is_same<boost::mpl::at<final_map, p_z>::type, range<-3,6,0,0,0,0>>::type::value),
                            "p_z range is wrong");
    GRIDTOOLS_STATIC_ASSERT((std::is_same<boost::mpl::at<final_map, p_i>::type, range<-6,10,0,0,0,0>>::type::value),
                            "p_i range is wrong");
    GRIDTOOLS_STATIC_ASSERT((std::is_same<boost::mpl::at<final_map, p_o>::type, range<0,0,0,0,0,0>>::type::value),
                            "p_o range is wrong");

    ASSERT_TRUE(true);
}
